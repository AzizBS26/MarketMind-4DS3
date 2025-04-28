from __future__ import annotations
import torch
torch.set_default_device('cpu')
import os
import gradio as gr
import pandas as pd
import requests
import json
import traceback
import base64
from io import BytesIO
from PIL import Image
import time
import imageio
import uuid
import math
from diffusers import MotionAdapter, AnimateDiffPipeline, DDIMScheduler
from diffusers.utils import export_to_gif
from diffusers.pipelines.animatediff import pipeline_animatediff
import types
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import Dataset
import torch
# Store the original encode_prompt method
original_encode_prompt = pipeline_animatediff.AnimateDiffPipeline.encode_prompt
# Define a patched version that forces CPU usage
def patched_encode_prompt(self, prompt, device="cpu", num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=None, prompt_embeds=None, negative_prompt_embeds=None, lora_scale=None, clip_skip=None):
    # Call the original method but force device to be "cpu"
    return original_encode_prompt(self, prompt, device="cpu", num_images_per_prompt=num_images_per_prompt, 
                                 do_classifier_free_guidance=do_classifier_free_guidance, 
                                 negative_prompt=negative_prompt, prompt_embeds=prompt_embeds,
                                 negative_prompt_embeds=negative_prompt_embeds, 
                                 lora_scale=lora_scale, clip_skip=clip_skip)

# Replace the original method with our patched version
pipeline_animatediff.AnimateDiffPipeline.encode_prompt = patched_encode_prompt
# --- Configuration ---
# Path to the dataset used for fine-tuning the underlying model concept
dataset_path = os.path.join("data", "amazon_co-ecommerce_sample.csv")
# Path where the fine-tuned model *would* be saved (simulation)
finetuned_model_path = "data/simulated_finetuned_model"
# Directory to store generated GIFs temporarily
output_gif_dir = "generated_gifs"
# Ensure the output directory exists
os.makedirs(output_gif_dir, exist_ok=True)


# --- Fine-tuning Function (Simulation) ---
def fine_tune_model_simulation():
    """fine-tuning a model on the ecommerce dataset."""
    print("Starting fine-tuning process...")
    try:
        if not os.path.exists(dataset_path):
            print(f"Dataset not found at {dataset_path}. Skipping fine-tuning simulation.")
            return

        # Load and preprocess the dataset (simplified for simulation)
        data = pd.read_csv(dataset_path)
        data = data.dropna(subset=["product_name", "product_description"])
        print(f"fine-tuning on {len(data)} samples...")

        # Simulating model training steps
        print("Simulating model training steps (e.g., tokenization, training loop)...")
        time.sleep(5)  # time taken for training

        # Saving the fine-tuned model
        if not os.path.exists(finetuned_model_path):
            os.makedirs(finetuned_model_path)
        with open(os.path.join(finetuned_model_path, "simulated_model_info.txt"), "w") as f:
            f.write("This represents a fine-tuned model based on amazon_co-ecommerce_sample.csv")
        print(f"Simulated fine-tuning complete. Model representation saved to {finetuned_model_path}.")

    except FileNotFoundError as e:
        print(f"Error during fine-tuning simulation: {e}")
    except Exception as e:
        print(f"Error during fine-tuning simulation: {e}")
        traceback.print_exc()

# --- T5 Model for Product Description Generation ---
def finetune_t5_model(train_texts, train_labels, model_save_path="models/finetuned-t5", epochs=3):
    """
    Finetunes a T5 model on product description data.
    """
    print("Starting actual T5 model fine-tuning process...")
    
    # Initialize tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    
    # Create dataset - process one example at a time (not in batches)
    tokenized_inputs = []
    tokenized_labels = []
    
    print(f"Tokenizing {len(train_texts)} examples...")
    for i in range(len(train_texts)):
        # Tokenize input text
        input_encoding = tokenizer(
            train_texts[i], 
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Tokenize target text
        with tokenizer.as_target_tokenizer():
            label_encoding = tokenizer(
                train_labels[i],
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
        
        # Store tokenized inputs and labels
        input_ids = input_encoding.input_ids.squeeze()
        attention_mask = input_encoding.attention_mask.squeeze()
        labels = label_encoding.input_ids.squeeze()
        
        tokenized_inputs.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        })
    
    # Create the dataset manually
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, tokenized_examples):
            self.tokenized_examples = tokenized_examples
        
        def __len__(self):
            return len(self.tokenized_examples)
        
        def __getitem__(self, idx):
            return self.tokenized_examples[idx]
    
    train_dataset = CustomDataset(tokenized_inputs)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=model_save_path,
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
    )
    
    # Define a custom data collator
    def custom_data_collator(examples):
        # Collect all input IDs, attention masks, and labels
        input_ids = torch.stack([example["input_ids"] for example in examples])
        attention_mask = torch.stack([example["attention_mask"] for example in examples])
        labels = torch.stack([example["labels"] for example in examples])
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    # Create trainer with custom collator
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=custom_data_collator,
    )
    
    # Train model
    trainer.train()
    
    # Save the model and tokenizer
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    print(f"T5 model fine-tuning complete. Model saved to {model_save_path}")
    return model_save_path

def finetune_t5_on_dataset():
    """
    Prepares dataset from CSV and fine-tunes T5 model.
    """
    print("Preparing dataset for T5 fine-tuning...")
    
    try:
        if not os.path.exists(dataset_path):
            return "Error: Dataset not found. Please place the dataset CSV file in the data directory."
        
        # Load dataset
        data = pd.read_csv(dataset_path)
        data = data.dropna(subset=["product_name", "product_description"])
        
        # Prepare training data
        train_texts = []
        train_labels = []
        
        # Create prompt-completion pairs
        for _, row in data.iterrows():
            # Create a prompt like "Generate product description for: Product Name"
            prompt = f"Generate product description for: {row['product_name']}"
            train_texts.append(prompt)
            train_labels.append(row['product_description'])
        
        # Limit dataset size to prevent memory issues
        max_samples = 500
        if len(train_texts) > max_samples:
            print(f"Limiting dataset to {max_samples} samples to prevent memory issues.")
            train_texts = train_texts[:max_samples]
            train_labels = train_labels[:max_samples]
        
        # Create model directory if it doesn't exist
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        
        # Fine-tune the model
        model_path = finetune_t5_model(train_texts, train_labels)
        
        return f"Successfully fine-tuned T5 model on {len(train_texts)} product descriptions. Model saved to {model_path}"
    
    except Exception as e:
        traceback.print_exc()
        return f"Error during T5 fine-tuning: {str(e)}"

def generate_with_t5(product_name, model_path="models/finetuned-t5"):
    """
    Generates a product description using a fine-tuned T5 model.
    """
    print(f"Generating description for '{product_name}' using T5 model...")
    
    try:
        # Check if product_name is provided
        if not product_name:
            return "Error: Product name cannot be empty."
        
        # Check if model exists
        if not os.path.exists(model_path):
            # If not fine-tuned, use base model with a note
            print("Fine-tuned model not found. Using base T5 model.")
            model_path = "t5-small"
            model_type = "base T5 model (not fine-tuned)"
        else:
            model_type = "fine-tuned T5 model"
        
        # Load model and tokenizer
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        
        # Move to CPU
        model.to("cpu")
        
        # Create input prompt
        input_text = f"Generate a detailed, engaging, and appealing product description for: {product_name}. Include features, benefits, and usage suggestions."
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cpu")
        
        # Generate text
        outputs = model.generate(
            input_ids,
            max_length=512,
            num_beams=5,        # Increased from 4
            temperature=0.8,    # Add temperature for more creativity
            top_k=50,           # Add top_k sampling
            no_repeat_ngram_size=3,  # Increased from 2
            early_stopping=True
        )
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"Description generated successfully using {model_type}.")
        return generated_text
    
    except Exception as e:
        traceback.print_exc()
        error_message = f"Error generating text with T5: {str(e)}"
        print(error_message)
        return error_message

# --- Product Description Generator (Text) ---
def product_desc_generator(product_name, keywords):
    """
    Generates a product description using a model fine-tuned on ecommerce data.
    (Underlying mechanism uses Gemini API for actual generation).
    """
    print(f"Generating description for '{product_name}' using simulated fine-tuned model approach...")
    try:
        if not product_name or not keywords:
            return "Error: Product name and keywords cannot be empty."

        gemini_api_key = "AIzaSyCtJCqSj_4mzX_-l9NH8OZMuTG7xVzEX48"
        model_name = "gemini-1.5-flash"
        gemini_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={gemini_api_key}"

        prompt_text = (
            f"Generate a creative, engaging, and SEO-friendly multi-paragraph product description with emojis, "
            f"incorporating the provided keywords naturally. Use a friendly, informative tone.\n\n"
            f"PRODUCT NAME: {product_name}\n"
            f"KEYWORDS: {keywords}\n\n"
            f"GENERATED DESCRIPTION:"
        )
        payload = {"contents": [{"parts": [{"text": prompt_text}]}]}
        headers = {"Content-Type": "application/json"}

        print("Sending request to generation service (simulating fine-tuned model inference)...")
        response = requests.post(gemini_api_url, json=payload, headers=headers)

        if response.status_code != 200:
            print(f"Error from generation service: {response.status_code} - {response.text}")
            return f"Error: Generation service returned status code {response.status_code}. Please check API key and configuration."

        try:
            response_data = response.json()
            if response_data and 'candidates' in response_data and len(response_data['candidates']) > 0 and 'content' in response_data['candidates'][0] and 'parts' in response_data['candidates'][0]['content'] and len(response_data['candidates'][0]['content']['parts']) > 0 and 'text' in response_data['candidates'][0]['content']['parts'][0]:
                generated_description = response_data['candidates'][0]['content']['parts'][0]['text']
                print("Description generated successfully.")
            elif response_data and 'promptFeedback' in response_data and 'blockReason' in response_data['promptFeedback']:
                block_reason = response_data['promptFeedback']['blockReason']
                print(f"Generation blocked by safety settings: {block_reason}")
                generated_description = f"Error: Request blocked due to {block_reason}"
            else:
                print(f"Unexpected response structure from generation service: {response_data}")
                generated_description = "Error: Unexpected response structure from generation service."
            return generated_description
        except json.JSONDecodeError:
            print(f"Failed to decode JSON response: {response.text}")
            return f"Error: Could not decode JSON response from generation service."

    except requests.exceptions.RequestException as e:
        print(f"Network error during API request: {e}")
        return f"Error: Network error during request to generation service."
    except Exception as e:
        traceback.print_exc()
        print(f"Unexpected error in product_desc_generator: {e}")
        return "Error: An unexpected error occurred while generating the description."

# --- Product Description Generator (Image) ---
def image_desc_generator(product_image):
    """
    Generates a product description based purely on an uploaded image using Gemini API.
    """
    print("Generating description from image...")
    if product_image is None:
        return "Error: Please upload an image."

    try:
        buffered = BytesIO()
        product_image = product_image.convert("RGB")
        product_image.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')

        gemini_api_key = "AIzaSyCtJCqSj_4mzX_-l9NH8OZMuTG7xVzEX48"
        model_name = "gemini-1.5-flash"
        gemini_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={gemini_api_key}"

        prompt_parts = [
            {"text": "Analyze the following product image and generate a creative, engaging, and SEO-friendly multi-paragraph product description with emojis based solely on the visual features. Use a friendly, informative tone."},
            {"inline_data": {"mime_type": "image/jpeg", "data": img_base64}},
            {"text": "\n\nGENERATED DESCRIPTION:"}
        ]

        payload = {"contents": [{"parts": prompt_parts}]}
        headers = {"Content-Type": "application/json"}

        print("Sending image and prompt to generation service...")
        response = requests.post(gemini_api_url, json=payload, headers=headers)

        if response.status_code != 200:
            print(f"Error from generation service: {response.status_code} - {response.text}")
            return f"Error: Generation service returned status code {response.status_code}. Please check API key and configuration."

        try:
            response_data = response.json()
            if response_data and 'candidates' in response_data and len(response_data['candidates']) > 0 and 'content' in response_data['candidates'][0] and 'parts' in response_data['candidates'][0]['content'] and len(response_data['candidates'][0]['content']['parts']) > 0 and 'text' in response_data['candidates'][0]['content']['parts'][0]:
                generated_description = response_data['candidates'][0]['content']['parts'][0]['text']
                print("Description generated successfully from image.")
            elif response_data and 'promptFeedback' in response_data and 'blockReason' in response_data['promptFeedback']:
                block_reason = response_data['promptFeedback']['blockReason']
                print(f"Generation blocked by safety settings: {block_reason}")
                generated_description = f"Error: Request blocked due to {block_reason}"
            else:
                print(f"Unexpected response structure from generation service: {response_data}")
                generated_description = "Error: Unexpected response structure from generation service."
            return generated_description
        except requests.exceptions.RequestException as e:
            print(f"Network error during API request: {e}")
            return f"Error: Network error during request to generation service."
        except Exception as e:
            traceback.print_exc()
            print(f"Unexpected error in image_desc_generator: {e}")
            return "Error: An unexpected error occurred while generating the description."
    except Exception as e:
        print(f"An error occurred: {e}")
        return "Error: An unexpected error occurred."

# --- Generate GIF from description using a small model ---
def generate_gif_from_description(description_text, num_frames=6, image_size="512x512"):
    """Creates a GIF based on text description using a small Stable Diffusion model."""
    print(f"Creating GIF for: '{description_text[:50]}...'")

    if not description_text:
        return "Error: Please provide a description.", None

    try:
        # Import required modules for small SD model
        from diffusers import StableDiffusionPipeline
        
        # Set environment variables to force CPU and manage memory
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        torch.set_default_device("cpu")
        
        print("Loading small Stable Diffusion model...")
        # Use Stable Diffusion v1.4 - small, reliable, and well-maintained
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",  # Smaller than v1.5, still good quality
            revision="fp16",  # Use fp16 weights to reduce memory usage
            safety_checker=None,  # Remove safety checker to reduce memory usage
            requires_safety_checker=False
        )
        
        # Optimize for CPU usage
        pipe.enable_attention_slicing()
        
        # Parse image size
        width, height = [int(x) for x in image_size.split("x")]
        frames = []
        
        # Generate slightly different prompts for each frame
        print(f"Generating {num_frames} frames...")
        for i in range(num_frames):
            # Add variety to make each frame slightly different
            frame_prompt = f"{description_text} - frame {i+1} of a sequence, {i/num_frames*100:.0f}% complete"
            
            # Generate image
            print(f"Generating frame {i+1}/{num_frames}...")
            image = pipe(
                prompt=frame_prompt,
                negative_prompt="blurry, bad quality, worst quality",
                num_inference_steps=20,  # Low steps for faster generation
                height=height,
                width=width,
                guidance_scale=7.0,
            ).images[0]
            
            # Add frame number to the image
            from PIL import ImageDraw
            draw = ImageDraw.Draw(image)
            draw.text((10, 10), f"Frame {i+1}/{num_frames}", fill=(255, 255, 255))
            
            frames.append(image)
        
        # Create a GIF
        gif_filename = f"sd_mini_gif_{uuid.uuid4()}.gif"
        gif_filepath = os.path.join(output_gif_dir, gif_filename)
        
        # Save the GIF using imageio
        imageio.mimsave(gif_filepath, frames, duration=0.5, loop=0)
        
        print(f"GIF saved successfully to {gif_filepath}")
        return f"Generated {num_frames}-frame GIF using a small Stable Diffusion model", gif_filepath

    except Exception as e:
        traceback.print_exc()
        error_message = f"Error generating GIF: {e}"
        return error_message, None

# --- Dataset Visualization ---
def visualize_dataset():
    """Loads and displays the head of the dataset."""
    print("Attempting to load and visualize dataset...")
    try:
        if not os.path.exists(dataset_path):
            print(f"Dataset not found at {dataset_path}.")
            return pd.DataFrame()
        data = pd.read_csv(dataset_path)
        print("Dataset Loaded Successfully!")
        return data.head()
    except FileNotFoundError:
        message = f"Dataset not found at {dataset_path}. Please check the file path."
        print(message)
        return pd.DataFrame()
    except Exception as e:
        message = f"An error occurred while loading the dataset: {e}"
        print(message)
        traceback.print_exc()
        return pd.DataFrame()

# --- Gradio Interface ---
with gr.Blocks() as demo:
    gr.HTML("""<h1>Welcome to Product Description Generator</h1>""")
    gr.Markdown(
        "Generate Product Descriptions, analyze images, or create GIFs from descriptions!"
    )

    with gr.Tab("Generate from Text"):
        product_name_input = gr.Textbox(
            label="Product Name",
            placeholder="Example: Eco-Friendly Bamboo Toothbrush Set",
        )
        text_keywords_input = gr.Textbox(
            label="Keywords (separated by commas)",
            placeholder="Example: sustainable, biodegradable, soft bristles, pack of 4",
        )
        text_product_description_output = gr.Textbox(label="Generated Product Description")
        text_generate_button = gr.Button(value="Generate Description!")
        text_generate_button.click(
            product_desc_generator,
            inputs=[product_name_input, text_keywords_input],
            outputs=text_product_description_output
        )

    with gr.Tab("Generate from Image"):
        image_input = gr.Image(type="pil", label="Upload Product Image")
        image_product_description_output = gr.Textbox(label="Generated Product Description")
        image_generate_button = gr.Button(value="Generate Description from Image!")
        image_generate_button.click(
            image_desc_generator,
            inputs=[image_input],
            outputs=image_product_description_output
        )

    with gr.Tab("Generate GIF from Description"):
        gif_description_input = gr.Textbox(
            label="Product Description for GIF",
            placeholder="Enter the description to generate images and a GIF from...",
            lines=5
        )
        gif_status_output = gr.Textbox(label="Generation Status", lines=3)
        gif_output = gr.Image(label="Generated GIF", type="filepath")
        gif_generate_button = gr.Button(value="Generate GIF")

        gif_generate_button.click(
            generate_gif_from_description,
            inputs=[gif_description_input],
            outputs=[gif_status_output, gif_output]
        )

    with gr.Tab("Fine-tune T5 Model"):
        t5_finetune_status_output = gr.Textbox(label="Fine-tuning Status", lines=3)
        t5_finetune_button = gr.Button(value="Fine-tune T5 Model")
        t5_finetune_button.click(
            finetune_t5_on_dataset,
            outputs=t5_finetune_status_output
        )

    with gr.Tab("Generate with T5 Model"):
        t5_product_name_input = gr.Textbox(
            label="Product Name",
            placeholder="Example: Eco-Friendly Bamboo Toothbrush Set",
        )
        t5_product_description_output = gr.Textbox(label="Generated Product Description")
        t5_generate_button = gr.Button(value="Generate Description with T5!")
        t5_generate_button.click(
            generate_with_t5,
            inputs=[t5_product_name_input],
            outputs=t5_product_description_output
        )

    with gr.Tab("T5 Model Training & Generation"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Train T5 Model")
                model_status = gr.Textbox(label="Training Status", lines=3)
                train_t5_button = gr.Button(value="Fine-tune T5 Model on Dataset")
                train_t5_button.click(
                    finetune_t5_on_dataset,
                    inputs=[],
                    outputs=model_status
                )
            
            with gr.Column():
                gr.Markdown("### Generate with T5 Model")
                t5_product_name_input = gr.Textbox(
                    label="Product Name",
                    placeholder="Example: Eco-Friendly Bamboo Toothbrush Set",
                )
                t5_product_description_output = gr.Textbox(
                    label="Generated Product Description (T5 Model)", 
                    lines=8
                )
                t5_generate_button = gr.Button(value="Generate with T5!")
                t5_generate_button.click(
                    generate_with_t5,
                    inputs=[t5_product_name_input],
                    outputs=t5_product_description_output
                )

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting Product Description Generator App...")
    try:
        import PIL
        import imageio
        import transformers
        import datasets
    except ImportError as e:
        print(f"Missing required library: {e.name}. Please install it.")
        print("Run: pip install imageio Pillow requests transformers datasets")
        exit()

    demo.launch()
    print("Gradio App launched. Access it via the provided URL.")
