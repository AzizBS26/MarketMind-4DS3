import os
import requests
import json
import traceback
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModelForSeq2SeqLM

def product_desc_generator(product_name, keywords, tone="Professional"):
    """
    Generates a product description using a model fine-tuned on ecommerce data.
    (Underlying mechanism uses Gemini API for actual generation).
    """
    print(f"Generating description for '{product_name}' using simulated fine-tuned model approach with {tone} tone...")
    try:
        if not product_name or not keywords:
            return "Error: Product name and keywords cannot be empty."

        gemini_api_key = "GEMINI_KEY"  # Replace with your actual Gemini API key
        model_name = "gemini-1.5-flash"
        gemini_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={gemini_api_key}"

        # Add tone to the prompt
        prompt_text = (
            f"Generate a creative, engaging, and SEO-friendly multi-paragraph product description with emojis, "
            f"incorporating the provided keywords naturally. Use a {tone.lower()} tone.\n\n"
            f"PRODUCT NAME: {product_name}\n"
            f"KEYWORDS: {keywords}\n"
            f"TONE: {tone}\n\n"
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

def generate_with_flan_t5(product_name, model_path="models/flan-t5-small"):
    """
    Generates a product description using a fine-tuned FLAN-T5-Small model.
    """
    print(f"Generating description for '{product_name}' using FLAN-T5-Small model...")
    
    try:
        # Check if product_name is provided
        if not product_name:
            return "Error: Product name cannot be empty."
        
        # Check if model exists
        if not os.path.exists(model_path):
            # If not fine-tuned, use base model with a note
            print("Fine-tuned FLAN-T5-Small model not found. Using base FLAN-T5-Small model.")
            model_path = "google/flan-t5-small"
            model_type = "base FLAN-T5-Small model (not fine-tuned)"
        else:
            model_type = "fine-tuned FLAN-T5-Small model"
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        
        # Move to CPU
        model.to("cpu")
        
        # Create input prompt
        input_text = f"Generate a detailed, engaging, and appealing product description for: {product_name}. Include features, benefits, and usage suggestions."
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cpu")
        
        # Generate text
        outputs = model.generate(
            input_ids,
            max_length=512,
            num_beams=5,
            temperature=0.8,
            top_k=50,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"Description generated successfully using {model_type}.")
        return generated_text
    
    except Exception as e:
        traceback.print_exc()
        error_message = f"Error generating text with FLAN-T5-Small: {str(e)}"
        print(error_message)
        return error_message