import os
from PIL import Image
import pandas as pd
from torchvision import transforms
import logging
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def tensor_to_pil(tensor):
    # Convert tensor to numpy array
    tensor = tensor.cpu().numpy()
    # Transpose from (C, H, W) to (H, W, C)
    tensor = tensor.transpose(1, 2, 0)
    # Denormalize
    tensor = (tensor * 0.5 + 0.5) * 255
    # Convert to uint8
    tensor = tensor.astype('uint8')
    # Convert to PIL Image
    return Image.fromarray(tensor)

def preprocess_images():
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Read metadata
    metadata_path = "images dataset/metadata.csv"
    images_dir = "images dataset/images"
    processed_dir = "images dataset/processed_images"
    
    # Create processed images directory
    os.makedirs(processed_dir, exist_ok=True)
    
    # Read metadata
    df = pd.read_csv(metadata_path)
    
    # Process each image
    for idx, row in df.iterrows():
        try:
            # Get the filename without the 'images/' prefix if it exists
            filename = row['filename']
            if filename.startswith('images/'):
                filename = filename[7:]  # Remove 'images/' prefix
            
            # Load image
            img_path = os.path.join(images_dir, filename)
            image = Image.open(img_path).convert("RGB")
            
            # Apply transformations
            tensor_image = transform(image)
            
            # Convert back to PIL Image
            processed_image = tensor_to_pil(tensor_image)
            
            # Save processed image
            save_path = os.path.join(processed_dir, filename)
            processed_image.save(save_path)
            
            logger.info(f"Processed image {idx + 1}/{len(df)}: {filename}")
            
        except Exception as e:
            logger.error(f"Error processing image {filename}: {str(e)}")
    
    logger.info("Image preprocessing completed")

if __name__ == "__main__":
    preprocess_images() 