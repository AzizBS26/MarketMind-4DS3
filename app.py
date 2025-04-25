from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import io
import base64
import logging
import numpy as np

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
model_id = "dreamlike-art/dreamlike-photoreal-2.0"  # Changed to a smaller model
try:
    logger.info("Initializing Stable Diffusion with GPU support")
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Initialize pipeline with GPU optimizations and robust download settings
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_safetensors=False,
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=False,
        resume_download=True,
        use_auth_token=False,
        trust_remote_code=True,
        revision="fp16"  # Use fp16 version for smaller size
    )
    
    # Move to GPU and enable optimizations
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    logger.info("Model loaded successfully on GPU")
except Exception as e:
    logger.error(f"Error initializing model: {str(e)}")
    raise

class PromptRequest(BaseModel):
    prompt: str
    negative_prompt: str = "ugly, blurry, poor quality, distorted, unrealistic, text, watermark, nsfw, inappropriate, adult content, violence, gore"

def process_image(image):
    """Process the generated image to ensure proper format and quality"""
    # Convert to numpy array
    img_array = np.array(image)
    
    # Log the range of values before processing
    logger.info(f"Image values range before processing: min={img_array.min()}, max={img_array.max()}")
    
    # Check if image is all black
    if img_array.max() == 0:
        logger.warning("Generated image is completely black, retrying with different parameters")
        return None
    
    # Ensure values are in valid range
    img_array = np.clip(img_array, 0, 1)
    
    # Convert back to PIL Image
    processed_image = Image.fromarray((img_array * 255).astype(np.uint8))
    
    # Verify the processed image
    processed_array = np.array(processed_image)
    logger.info(f"Processed image values range: min={processed_array.min()}, max={processed_array.max()}")
    
    return processed_image

@app.post("/generate-flyer")
async def generate_flyer(request: PromptRequest):
    try:
        logger.info(f"Generating image with prompt: {request.prompt}")
        
        # Add business-specific keywords to the prompt
        enhanced_prompt = f"photorealistic style, {request.prompt}, professional business design, corporate style, clean layout, business appropriate, family friendly, work safe, high quality, detailed, 4k, vibrant colors, well lit, professional photography, studio lighting"
        
        # Generate image with optimized settings
        image = pipe(
            prompt=enhanced_prompt,
            negative_prompt=request.negative_prompt,
            num_inference_steps=25,  # Reduced steps for faster generation
            guidance_scale=7.5,
            width=512,
            height=512,
            num_images_per_prompt=1
        ).images[0]

        # Process the image
        processed_image = process_image(image)
        
        if processed_image is None:
            # Retry with different parameters
            logger.info("Retrying image generation with different parameters")
            image = pipe(
                prompt=enhanced_prompt,
                negative_prompt=request.negative_prompt,
                num_inference_steps=40,
                guidance_scale=8.5,
                width=512,
                height=512,
                num_images_per_prompt=1
            ).images[0]
            processed_image = process_image(image)

        if processed_image is None:
            raise HTTPException(status_code=500, detail="Failed to generate valid image after multiple attempts")

        # Convert image to base64 with explicit format
        buffered = io.BytesIO()
        processed_image.save(buffered, format="PNG", quality=100)
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Clear GPU cache after generation
        torch.cuda.empty_cache()
        
        logger.info("Image generated and processed successfully")
        return {"image": img_str}
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000) 