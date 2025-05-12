import os
import torch
import uuid
import traceback
import imageio
from PIL import ImageDraw
from diffusers import StableDiffusionPipeline

def generate_gif_from_description(description_text, num_frames=6, image_size="512x512", style="product"):
    """Creates a GIF based on text description using a small Stable Diffusion model."""
    print(f"Creating GIF for: '{description_text[:50]}...'")

    if not description_text:
        return "Error: Please provide a description.", None

    try:
        # Set environment variables to force CPU and manage memory
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        torch.set_default_device("cpu")
        
        # Set Hugging Face cache directory to D: drive
        os.environ["HF_HOME"] = "D:/HuggingFace"
        os.environ["TRANSFORMERS_CACHE"] = "D:/HuggingFace/transformers"
        os.environ["HF_DATASETS_CACHE"] = "D:/HuggingFace/datasets"
        
        print("Loading small Stable Diffusion model (downloading to D: drive)...")
        
        # Create cache directories if they don't exist
        os.makedirs("D:/HuggingFace", exist_ok=True)
        os.makedirs("D:/HuggingFace/transformers", exist_ok=True)
        os.makedirs("D:/HuggingFace/datasets", exist_ok=True)
        
        # Use Stable Diffusion v1.4 - small, reliable, and well-maintained
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            revision="fp16",
            safety_checker=None,
            requires_safety_checker=False,
            cache_dir="D:/HuggingFace"  # Explicit cache directory
        )
        
        # Optimize for CPU usage
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()  # Additional optimization
        
        # Parse image size
        width, height = [int(x) for x in image_size.split("x")]
        frames = []
        
        # Style enhancement templates
        style_templates = {
            "product": "professional product photography, {}, studio lighting, detailed, 8k, commercial photography",
            "realistic": "photorealistic {}, detailed texture, natural lighting, 8k resolution",
            "artistic": "artistic rendering of {}, colorful, creative composition, vibrant colors",
            "minimalist": "minimalist design of {}, clean background, simple lines, elegant, modern",
            "isometric": "isometric view of {}, 3D rendering, clean design, product visualization"
        }
        
        # Apply style template
        style_prompt = style_templates.get(style, style_templates["product"]).format(description_text)
        
        # Generate seed for consistency between frames
        generator = torch.Generator().manual_seed(42)  # Fixed seed for consistency
        
        # Animation parameters
        zoom_factors = [1.0 + (i * 0.05) for i in range(num_frames)]  # Gradual zoom effect
        
        # Generate frames with consistent style but subtle changes
        print(f"Generating {num_frames} frames...")
        
        # First, generate a base image with more steps for quality
        print("Generating base image...")
        base_image = pipe(
            prompt=style_prompt,
            negative_prompt="blurry, bad quality, worst quality, text, watermark",
            num_inference_steps=30,
            height=height,
            width=width,
            guidance_scale=7.5,
            generator=generator
        ).images[0]
        
        # Now generate variations for animation
        frames.append(base_image)
        
        for i in range(1, num_frames):
            print(f"Generating frame {i+1}/{num_frames}...")
            
            # For animation consistency, use fewer steps but same seed family
            frame_generator = torch.Generator().manual_seed(42 + i)
            
            # Slight prompt variation for subtle differences
            frame_prompt = f"{style_prompt}, {['slightly rotated', 'different angle', 'zoomed in', 'alternative view'][i % 4]}"
            
            image = pipe(
                prompt=frame_prompt,
                negative_prompt="blurry, bad quality, worst quality, text, watermark",
                num_inference_steps=20,  # Fewer steps for variation frames
                height=height,
                width=width,
                guidance_scale=7.5,
                generator=frame_generator
            ).images[0]
            
            # Apply zoom effect (crop and resize)
            if zoom_factors[i] > 1.0:
                # Calculate crop dimensions
                new_width = int(width / zoom_factors[i])
                new_height = int(height / zoom_factors[i])
                left = (width - new_width) // 2
                top = (height - new_height) // 2
                right = left + new_width
                bottom = top + new_height
                
                # Crop and resize
                image = image.crop((left, top, right, bottom)).resize((width, height))
            
            frames.append(image)
        
        # Create a GIF with smoother animation
        output_gif_dir = "generated_gifs"
        os.makedirs(output_gif_dir, exist_ok=True)
        gif_filename = f"enhanced_gif_{uuid.uuid4()}.gif"
        gif_filepath = os.path.join(output_gif_dir, gif_filename)
        
        # Create forward and backward frames for a smoother loop
        loop_frames = frames + frames[-2:0:-1]  # Add reversed frames (excluding first and last)
        
        # Save with better settings
        imageio.mimsave(
            gif_filepath, 
            loop_frames, 
            duration=0.2,  # Faster animation (5fps)
            loop=0,        # Infinite loop
            optimize=True  # Optimize file size
        )
        
        print(f"Enhanced GIF saved successfully to {gif_filepath}")
        return f"Generated {len(loop_frames)}-frame enhanced GIF with zoom effects", gif_filepath

    except Exception as e:
        traceback.print_exc()
        error_message = f"Error generating GIF: {e}"
        return error_message, None