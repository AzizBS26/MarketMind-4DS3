import os
import pandas as pd
import torch
from torch.utils.data import Dataset

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_examples):
        self.tokenized_examples = tokenized_examples
    
    def __len__(self):
        return len(self.tokenized_examples)
    
    def __getitem__(self, idx):
        return self.tokenized_examples[idx]

def visualize_dataset(dataset_path):
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
        import traceback
        traceback.print_exc()
        return pd.DataFrame()