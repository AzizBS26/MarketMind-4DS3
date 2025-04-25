import os
import torch
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, AutoencoderKL
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
import pandas as pd
from PIL import Image
import logging
from tqdm.auto import tqdm
import numpy as np
import time
from huggingface_hub import HfFolder
from huggingface_hub.utils import RepositoryNotFoundError
from torchvision import transforms
import platform
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_system_requirements():
    """Check system requirements and provide appropriate warnings"""
    if platform.system() == "Windows":
        logger.info("Running on Windows - some optimizations will be disabled")
        logger.info("Note: Triton is not available on Windows systems")
    else:
        try:
            import triton
            logger.info("Triton is available - optimizations enabled")
        except ImportError:
            logger.warning("Triton is not available. Some optimizations will be disabled.")
            logger.info("To install Triton, run: pip install triton")

def setup_gpu():
    """Setup GPU memory management"""
    if torch.cuda.is_available():
        # Set memory allocation strategy
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        gc.collect()
        
        # Get GPU info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"Using GPU: {gpu_name} ({gpu_memory:.2f} GB)")
        
        return True
    else:
        logger.error("No GPU available. Training requires a GPU.")
        return False

def download_with_retry(model_id, max_retries=3, retry_delay=5):
    """Download model with retry logic"""
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries} to download model")
            # Increase timeout for downloads
            os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '600'  # 10 minutes
            os.environ['HF_HUB_OFFLINE'] = '0'
            
            # Try to download the model
            tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
            text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
            vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float32,  # Use float32 for training
                use_safetensors=False,
                safety_checker=None
            )
            return tokenizer, text_encoder, vae, pipe
            
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"Waiting {retry_delay} seconds before retrying...")
                time.sleep(retry_delay)
            else:
                raise Exception(f"Failed to download model after {max_retries} attempts")

class CustomDataset(Dataset):
    def __init__(self, metadata_path, images_dir, tokenizer, transform=None):
        self.metadata = pd.read_csv(metadata_path)
        self.images_dir = images_dir
        self.tokenizer = tokenizer
        
        # Define default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        else:
            self.transform = transform
        
        # Verify paths exist
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        
        # Log dataset statistics
        logger.info(f"Loaded dataset with {len(self.metadata)} samples")
        logger.info(f"Images directory: {images_dir}")
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Get the filename without the 'images/' prefix if it exists
        filename = row['filename']
        if filename.startswith('images/'):
            filename = filename[7:]  # Remove 'images/' prefix
        
        # Construct image path
        img_path = os.path.join(self.images_dir, filename)
        
        # Verify file exists
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        
        try:
            # Load and transform image
            image = Image.open(img_path).convert("RGB")
            image_tensor = self.transform(image)
            
            # Tokenize text
            text_inputs = self.tokenizer(
                row['text'],
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            
            return {
                "pixel_values": image_tensor,
                "input_ids": text_inputs.input_ids[0],
                "text": row['text']  # Add the original text for debugging
            }
            
        except Exception as e:
            logger.error(f"Error processing image {img_path}: {str(e)}")
            raise

def train_model():
    try:
        # Check system requirements
        check_system_requirements()
        
        # Setup GPU
        if not setup_gpu():
            return
        
        # Initialize the model with retry logic
        model_id = "runwayml/stable-diffusion-v1-5"
        tokenizer, text_encoder, vae, pipe = download_with_retry(model_id)
        
        # Move to GPU and ensure consistent dtype
        pipe = pipe.to("cuda")
        text_encoder = text_encoder.to("cuda")
        vae = vae.to("cuda")
        
        # Enable gradient checkpointing to save memory
        pipe.unet.enable_gradient_checkpointing()
        
        # Ensure all components use float32
        pipe.unet = pipe.unet.float()
        text_encoder = text_encoder.float()
        vae = vae.float()
        
        # Create dataset
        dataset = CustomDataset(
            metadata_path="images dataset/metadata.csv",
            images_dir="images dataset/images",
            tokenizer=tokenizer
        )
        
        # Create dataloader with appropriate number of workers
        num_workers = 0 if platform.system() == "Windows" else 4
        train_dataloader = DataLoader(
            dataset,
            batch_size=2,  # Reduced batch size to save memory
            shuffle=True,
            num_workers=num_workers
        )
        
        # Training parameters
        num_epochs = 10
        learning_rate = 1e-5
        optimizer = torch.optim.AdamW(
            pipe.unet.parameters(),
            lr=learning_rate
        )
        
        # Learning rate scheduler
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_dataloader) * num_epochs
        )
        
        # Training loop
        for epoch in range(num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
            progress_bar = tqdm(total=len(train_dataloader))
            
            for batch in train_dataloader:
                # Clear GPU cache
                torch.cuda.empty_cache()
                
                # Move batch to GPU and ensure float32
                pixel_values = batch["pixel_values"].to("cuda").float()
                input_ids = batch["input_ids"].to("cuda")
                texts = batch["text"]  # Get the original texts
                
                # Get text embeddings
                with torch.no_grad():
                    text_embeddings = text_encoder(input_ids)[0].float()
                
                # Encode images to latents
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                
                # Add noise to latents
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
                
                # Forward pass
                noise_pred = pipe.unet(noisy_latents, timesteps, text_embeddings).sample
                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                # Clear memory
                del noise_pred, loss
                torch.cuda.empty_cache()
                
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": loss.item()})
            
            # Save checkpoint
            save_path = f"checkpoints/epoch_{epoch + 1}"
            os.makedirs(save_path, exist_ok=True)
            pipe.save_pretrained(save_path)
            logger.info(f"Saved checkpoint to {save_path}")
            
            # Clear memory after epoch
            torch.cuda.empty_cache()
            gc.collect()
        
        # Save final model
        pipe.save_pretrained("trained_model")
        logger.info("Training completed and model saved")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    train_model() 