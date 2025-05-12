import os
import traceback
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments

from utils.dataset import CustomDataset

def finetune_flan_t5_model(train_texts, train_labels, model_save_path="models/flan-t5-small", epochs=3):
    """
    Finetunes a FLAN-T5-Small model on product description data.
    """
    print("Starting actual FLAN-T5-Small model fine-tuning process...")
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    
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
    
    # Use your existing CustomDataset class
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
    
    # Use your existing custom_data_collator
    def custom_data_collator(examples):
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
    
    print(f"FLAN-T5-Small model fine-tuning complete. Model saved to {model_save_path}")
    return model_save_path

def finetune_flan_t5_on_dataset(dataset_path):
    """
    Prepares dataset from CSV and fine-tunes FLAN-T5-Small model.
    """
    print("Preparing dataset for FLAN-T5-Small fine-tuning...")
    
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
        
        print(f"Training on full dataset with {len(train_texts)} examples")
        
        # Create model directory if it doesn't exist
        model_dir = "models/flan-t5-small"
        os.makedirs(model_dir, exist_ok=True)
        
        # Fine-tune the model
        model_path = finetune_flan_t5_model(train_texts, train_labels)
        
        return f"Successfully fine-tuned FLAN-T5-Small model on {len(train_texts)} product descriptions. Model saved to {model_path}"
    
    except Exception as e:
        traceback.print_exc()
        return f"Error during FLAN-T5 fine-tuning: {str(e)}"