import os
import traceback
import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments

from utils.dataset import CustomDataset

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
    train_dataset = CustomDataset(tokenized_inputs)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=model_save_path,
        learning_rate=5e-5,
        per_device_train_batch_size=2,  # Reduced from 4 to use less memory
        gradient_accumulation_steps=4,  # Added to effectively use larger batches
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

def finetune_t5_on_dataset(dataset_path):
    """
    Prepares dataset from CSV and fine-tunes T5 model.
    """
    print("Preparing dataset for T5 fine-tuning...")
    
    try:
        if not os.path.exists(dataset_path):
            return "Error: Dataset not found. Please place the dataset CSV file in the data directory."
        
        # Load dataset
        data = pd.read_csv(dataset_path)
        
        # Update these to match your actual column names in the CSV
        input_text_column = "input_text"  # Instead of "product_name"
        target_text_column = "target_text"  # Instead of "product_description"
        
        data = data.dropna(subset=[input_text_column, target_text_column])
        
        # Prepare training data
        train_texts = []
        train_labels = []
        
        # Create prompt-completion pairs
        for _, row in data.iterrows():
            # The input text already has the "generate description:" prefix
            train_texts.append(row[input_text_column])
            train_labels.append(row[target_text_column])
        
        print(f"Training on full dataset with {len(train_texts)} examples")
        
        # Create model directory if it doesn't exist
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        
        # Fine-tune the model
        model_path = finetune_t5_model(train_texts, train_labels)
        
        return f"Successfully fine-tuned T5 model on {len(train_texts)} product descriptions. Model saved to {model_path}"
    
    except Exception as e:
        traceback.print_exc()
        return f"Error during T5 fine-tuning: {str(e)}"