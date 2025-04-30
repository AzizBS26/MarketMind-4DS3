from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import logging
import numpy as np
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the Stable Diffusion pipeline
model_id = "runwayml/stable-diffusion-v1-5"  # Changed to a more reliable model
try:
    logger.info("Initializing Stable Diffusion with GPU support")
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Adjust the Stable Diffusion pipeline initialization to remove unsupported parameters and ensure compatibility
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,
        local_files_only=False,
        resume_download=True
    )

    # Add a fallback to CPU if GPU initialization fails
    try:
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()
        torch.cuda.empty_cache()
        logger.info("Model loaded successfully on GPU")
    except Exception as gpu_error:
        logger.warning(f"GPU initialization failed: {gpu_error}. Falling back to CPU.")
        pipe = pipe.to("cpu")
        logger.info("Model loaded successfully on CPU")

except Exception as e:
    logger.error(f"Error initializing model: {str(e)}")
    raise

class PromptRequest(BaseModel):
    prompt: str
    negative_prompt: str = "blurry, low quality, distorted, overexposed"

def process_image(image):
    """Process the generated image to ensure proper format and quality"""
    # Convert to numpy array
    img_array = np.array(image)

    # Log the range of values before processing
    logger.info(f"Image values range before processing: min={img_array.min()}, max={img_array.max()}")

    # Check if image contains invalid values (e.g., NaN or infinity)
    if not np.isfinite(img_array).all():
        logger.warning("Image contains invalid values (NaN or infinity). Replacing with zeros.")
        img_array = np.nan_to_num(img_array, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure values are in valid range [0, 1]
    img_array = np.clip(img_array / 255.0, 0, 1)  # Normalize values to [0, 1]

    # Convert back to PIL Image
    processed_image = Image.fromarray((img_array * 255).astype(np.uint8))

    # Verify the processed image
    processed_array = np.array(processed_image)
    logger.info(f"Processed image values range: min={processed_array.min()}, max={processed_array.max()}")

    return processed_image

# Function to add text to an image
def add_text_to_image(image, text, position=(10, 10)):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", size=30)
    draw.text(position, text, fill="white", font=font)
    return image

# Update to handle missing model more gracefully
@app.on_event("startup")
def check_model_completeness():
    try:
        # Check if the required model file exists
        model_path = "unet/diffusion_pytorch_model.safetensors"  # Adjusted to check for the specific file
        if not os.path.exists(model_path):
            logger.error("The required model.safetensors file is missing. Please ensure the model is complete.")
            logger.info("To download the complete model, visit: https://huggingface.co/runwayml/stable-diffusion-v1-5")
            logger.info("The application will start, but image generation will be disabled until the model is complete.")
            return False
        logger.info("Model completeness verified.")
        return True
    except Exception as e:
        logger.error(f"Unexpected error during model check: {str(e)}")
        return False

# Ensure the `pipe` variable is properly initialized and globally accessible
@app.post("/generate-flyer")
async def generate_flyer(request: PromptRequest):
    global pipe  # Declare `pipe` as global to ensure it is accessible

    try:
        logger.info(f"Generating image with prompt: {request.prompt}")

        # Simplify the prompt for basic image generation
        enhanced_prompt = f"{request.prompt}, vibrant colors, detailed, high quality flyer design"

        processed_image = None
        use_gpu = torch.cuda.is_available()  # Check if GPU is available

        for attempt in range(3):
            try:
                # Generate image with adjusted settings
                image = pipe(
                    prompt=enhanced_prompt,
                    negative_prompt=request.negative_prompt,
                    num_inference_steps=100,  # Moderate steps for better results
                    guidance_scale=10,  # Standard guidance scale
                    width=768,  # Standard resolution
                    height=768,  # Standard resolution
                    num_images_per_prompt=1
                ).images[0]

                # Process the image
                processed_image = process_image(image)

                # Check if the processed image is valid (not black)
                img_array = np.array(processed_image)
                if img_array.min() == 0 and img_array.max() == 0:
                    logger.warning("Generated image is completely black. Retrying with reinitialized pipeline.")

                    # Reinitialize the pipeline
                    pipe.to("cpu")  # Switch to CPU to avoid GPU memory issues
                    pipe = StableDiffusionPipeline.from_pretrained(
                        model_id,
                        torch_dtype=torch.float32,
                        safety_checker=None,
                        local_files_only=False,
                        resume_download=True
                    )
                    if use_gpu:
                        pipe.to("cuda")
                    continue

                # If the image is valid, break the loop
                break
            except Exception as gen_error:
                logger.error(f"Error during image generation attempt {attempt + 1}: {gen_error}")

                # If all GPU attempts fail, switch to CPU
                if attempt == 2 and use_gpu:
                    logger.warning("Switching to CPU for image generation")
                    pipe.to("cpu")
                    use_gpu = False

        if processed_image is None:
            raise HTTPException(status_code=500, detail="Failed to generate valid image after multiple attempts")

        # Convert image to base64 with explicit format
        buffered = io.BytesIO()
        processed_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Clear GPU cache after generation if GPU was used
        if use_gpu:
            torch.cuda.empty_cache()

        logger.info("Image generated and processed successfully")
        return {"image": img_str}
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Modify the main function to disable image generation if the model is incomplete
if __name__ == "__main__":
    model_ready = check_model_completeness()
    if not model_ready:
        logger.warning("Model is incomplete. Image generation features will be disabled.")
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)