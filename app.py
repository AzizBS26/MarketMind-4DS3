from __future__ import annotations
import torch
torch.set_default_device('cpu')
import os
import gradio as gr
import types
from diffusers.pipelines.animatediff import pipeline_animatediff

# Import components
from utils.dataset import visualize_dataset
from generators.text_generator import product_desc_generator, generate_with_t5, generate_with_flan_t5
from generators.image_generator import image_desc_generator, generate_product_image
from generators.gif_generator import generate_gif_from_description
from trainers.t5_trainer import finetune_t5_on_dataset
from trainers.flan_t5_trainer import finetune_flan_t5_on_dataset

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
dataset_path = os.path.join("data", "t5_product_description_data.csv")
# Path where the fine-tuned model *would* be saved (simulation)
finetuned_model_path = "data/simulated_finetuned_model"
# Directory to store generated GIFs temporarily
output_gif_dir = "generated_gifs"
# Ensure the output directory exists
os.makedirs(output_gif_dir, exist_ok=True)
os.makedirs("models", exist_ok=True)

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
        tone_choices = gr.Dropdown(
            label="Select Tone",
            choices=["Professional", "Casual", "Enthusiastic", "Luxury", "Technical", "Minimalist"], 
            value="Professional"
        )
        text_product_description_output = gr.Textbox(label="Generated Product Description")
        text_generate_button = gr.Button(value="Generate Description!")
        text_generate_button.click(
            product_desc_generator,
            inputs=[product_name_input, text_keywords_input, tone_choices],
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
            lambda: finetune_t5_on_dataset(dataset_path),
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

    with gr.Tab("Fine-tune FLAN-T5 Model"):
        flan_t5_finetune_status_output = gr.Textbox(label="Fine-tuning Status", lines=3)
        flan_t5_finetune_button = gr.Button(value="Fine-tune FLAN-T5 Model")
        flan_t5_finetune_button.click(
            lambda: finetune_flan_t5_on_dataset(dataset_path),
            outputs=flan_t5_finetune_status_output
        )

    with gr.Tab("Generate with FLAN-T5 Model"):
        flan_t5_product_name_input = gr.Textbox(
            label="Product Name",
            placeholder="Example: Eco-Friendly Bamboo Toothbrush Set",
        )
        flan_t5_product_description_output = gr.Textbox(label="Generated Product Description")
        flan_t5_generate_button = gr.Button(value="Generate Description with FLAN-T5!")
        flan_t5_generate_button.click(
            generate_with_flan_t5,
            inputs=[flan_t5_product_name_input],
            outputs=flan_t5_product_description_output
        )

    with gr.Tab("FLAN-T5 Model"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Train FLAN-T5-Small Model")
                flan_t5_model_status = gr.Textbox(label="Training Status", lines=3)
                flan_t5_train_button = gr.Button(value="Fine-tune FLAN-T5-Small Model")
                flan_t5_train_button.click(
                    lambda: finetune_flan_t5_on_dataset(dataset_path),
                    outputs=flan_t5_model_status
                )
            
            with gr.Column():
                gr.Markdown("### Generate with FLAN-T5-Small Model")
                flan_t5_product_name_input = gr.Textbox(
                    label="Product Name",
                    placeholder="Example: Eco-Friendly Bamboo Toothbrush Set",
                )
                flan_t5_product_description_output = gr.Textbox(
                    label="Generated Product Description (FLAN-T5-Small)", 
                    lines=8
                )
                flan_t5_generate_button = gr.Button(value="Generate with FLAN-T5!")
                flan_t5_generate_button.click(
                    generate_with_flan_t5,
                    inputs=[flan_t5_product_name_input],
                    outputs=flan_t5_product_description_output
                )

    with gr.Tab("T5 Model Training & Generation"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Train T5 Model")
                model_status = gr.Textbox(label="Training Status", lines=3)
                train_t5_button = gr.Button(value="Fine-tune T5 Model on Dataset")
                train_t5_button.click(
                    lambda: finetune_t5_on_dataset(dataset_path),
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

    with gr.Tab("Generate Product Image"):
        gr.Markdown("### Generate Product Images from Keywords")
        
        with gr.Row():
            with gr.Column():
                product_name_input = gr.Textbox(
                    label="Product Name", 
                    placeholder="Enter product name (e.g., Wireless Gaming Mouse)",
                    lines=1
                )
                product_keywords = gr.Textbox(
                    label="Additional Keywords (optional)",
                    placeholder="gaming, RGB lighting, ergonomic, black",
                    lines=2
                )
                image_style = gr.Radio(
                    choices=["product", "realistic", "artistic", "minimalist", "isometric"],
                    label="Image Style",
                    value="product"
                )
                image_size = gr.Radio(
                    choices=["512x512", "640x384", "384x640", "768x512"],
                    label="Image Size",
                    value="512x512"
                )
                gen_image_btn = gr.Button("Generate Product Image", variant="primary")
            
            with gr.Column():
                image_output = gr.Image(label="Generated Product Image")
                image_status = gr.Textbox(label="Status", interactive=False)
        
        gen_image_btn.click(
            fn=generate_product_image,
            inputs=[product_name_input, product_keywords, image_style, image_size],
            outputs=[image_status, image_output]
        )

# --- Main Execution ---
if __name__ == "__main__":
    # Create required directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("generated_gifs", exist_ok=True)
    os.makedirs("generated_images", exist_ok=True)  # Add this line
    
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
