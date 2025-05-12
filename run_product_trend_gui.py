#!/usr/bin/env python3
"""
Launcher script for Product Trend Prediction GUI
"""
import os
import sys

# Change to the script's directory to ensure relative paths work
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Import the GUI class
from product_trend_gui import ProductTrendGUI
import tkinter as tk

if __name__ == "__main__":
    print("Starting Product Trend Prediction GUI...")
    print(f"Working directory: {os.getcwd()}")
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(script_dir, "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created data directory: {data_dir}")
    
    # Create and run the GUI
    root = tk.Tk()
    app = ProductTrendGUI(root)
    root.mainloop() 