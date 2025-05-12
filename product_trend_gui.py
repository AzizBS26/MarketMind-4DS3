#!/usr/bin/env python3
"""
Product Trend Prediction GUI

A graphical interface for predicting which products will trend based on client inputs.
"""
import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use("TkAgg")

# Ensure correct path resolution
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Import the product trend predictor
from product_trend_prediction import ProductTrendPredictor

class ProductTrendGUI:
    """GUI for product trend prediction"""
    
    def __init__(self, root):
        """Initialize the GUI"""
        self.root = root
        self.root.title("MarketMind - Product Trend Predictor")
        self.root.geometry("1200x800")  # Increased width for better visibility
        self.root.minsize(1100, 750)    # Increased minimum size
        
        # Set color scheme
        self.bg_color = "#f8f8f8"
        self.primary_color = "#2979FF"
        self.secondary_color = "#FF5722"
        self.success_color = "#4CAF50"
        self.warning_color = "#FFC107"
        self.text_color = "#212121"
        self.light_text = "#757575"
        
        # Configure font styles
        self.title_font = ("Segoe UI", 20, "bold")
        self.header_font = ("Segoe UI", 16, "bold")
        self.subheader_font = ("Segoe UI", 14)
        self.body_font = ("Segoe UI", 12)
        self.small_font = ("Segoe UI", 10)
        
        # Configure styles
        self.style = ttk.Style()
        self.style.configure("TFrame", background=self.bg_color)
        self.style.configure("TLabel", background=self.bg_color, foreground=self.text_color, font=self.body_font)
        self.style.configure("TButton", font=self.body_font)
        self.style.configure("Title.TLabel", font=self.title_font, background=self.bg_color)
        self.style.configure("Header.TLabel", font=self.header_font, background=self.bg_color)
        self.style.configure("Subheader.TLabel", font=self.subheader_font, background=self.bg_color)
        self.style.configure("Card.TFrame", background="white", relief="ridge", borderwidth=1)
        
        # Initialize the product trend predictor
        try:
            self.predictor = ProductTrendPredictor()
            self.initialized = True
        except Exception as e:
            self.initialized = False
            messagebox.showerror("Initialization Error", f"Failed to initialize predictor: {e}")
        
        # Create the UI
        self.create_widgets()
    
    def create_widgets(self):
        """Create the GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, padding=20, style="TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = ttk.Frame(main_frame, style="TFrame")
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        title = ttk.Label(header_frame, text="Product Trend Predictor", style="Title.TLabel")
        title.pack(anchor=tk.CENTER)
        
        subtitle = ttk.Label(header_frame, 
                            text="Discover which products will trend in the future based on your criteria",
                            foreground=self.light_text)
        subtitle.pack(anchor=tk.CENTER)
        
        # Create horizontal paned window for main content
        self.paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Left pane for inputs
        left_pane = ttk.Frame(self.paned_window, style="TFrame")
        
        # Client Input Frame
        input_frame = ttk.LabelFrame(left_pane, text="Your Search Criteria", padding=15)
        input_frame.pack(fill=tk.X, pady=10)
        
        # Business type and demographics
        self.create_input_section(input_frame)
        
        # Search button frame
        button_frame = ttk.Frame(left_pane, style="TFrame")
        button_frame.pack(fill=tk.X, pady=15)
        
        search_button = ttk.Button(button_frame, text="Predict Trending Products", 
                                  command=self.search_products, padding=10)
        search_button.pack(side=tk.LEFT, padx=5)
        
        # Visualization frame (initially empty)
        self.viz_container = ttk.LabelFrame(left_pane, text="Trend Visualization", padding=10)
        self.viz_container.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Add left pane to paned window
        self.paned_window.add(left_pane, weight=3)
        
        # Right pane for results
        right_pane = ttk.Frame(self.paned_window, style="TFrame")
        
        # Results frame
        self.results_frame = ttk.LabelFrame(right_pane, text="Predicted Trending Products", padding=15)
        self.results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Initial empty state message
        empty_msg = ttk.Label(self.results_frame, 
                             text="Enter your criteria and click 'Predict Trending Products' to see results",
                             foreground=self.light_text)
        empty_msg.pack(pady=50)
        
        # Add right pane to paned window
        self.paned_window.add(right_pane, weight=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W, padding=5)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=(5, 0))
    
    def create_input_section(self, parent):
        """Create the input form section"""
        # Create grid for inputs
        form_frame = ttk.Frame(parent, style="TFrame")
        form_frame.pack(fill=tk.X, pady=5)
        
        # Budget range
        budget_frame = ttk.Frame(form_frame, style="TFrame")
        budget_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(budget_frame, text="Price Range:").pack(side=tk.LEFT, padx=(0, 10))
        
        min_frame = ttk.Frame(budget_frame, style="TFrame")
        min_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Label(min_frame, text="Min $").pack(side=tk.LEFT)
        self.min_price_var = tk.StringVar(value="10")
        min_price = ttk.Entry(min_frame, textvariable=self.min_price_var, width=8)
        min_price.pack(side=tk.LEFT, padx=5)
        
        max_frame = ttk.Frame(budget_frame, style="TFrame")
        max_frame.pack(side=tk.LEFT)
        
        ttk.Label(max_frame, text="Max $").pack(side=tk.LEFT)
        self.max_price_var = tk.StringVar(value="500")
        max_price = ttk.Entry(max_frame, textvariable=self.max_price_var, width=8)
        max_price.pack(side=tk.LEFT, padx=5)
        
        # Categories section
        categories_frame = ttk.Frame(form_frame, style="TFrame")
        categories_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(categories_frame, text="Product Categories:").pack(anchor=tk.W, pady=(0, 5))
        
        # Categories grid
        cat_grid = ttk.Frame(categories_frame, style="TFrame")
        cat_grid.pack(fill=tk.X)
        
        # List of product categories
        categories = [
            "Clothing", "Electronics", "Home Decor", "Beauty", "Health", 
            "Fitness", "Kitchen", "Office", "Outdoors", "Pet Supplies",
            "Food & Beverage", "Toys", "Books", "Art Supplies", "Handmade"
        ]
        
        # Create category checkboxes
        self.category_vars = {}
        for i, category in enumerate(categories):
            row, col = i // 5, i % 5
            var = tk.BooleanVar(value=False)
            self.category_vars[category] = var
            cb = ttk.Checkbutton(cat_grid, text=category, variable=var)
            cb.grid(row=row, column=col, sticky=tk.W, padx=5, pady=2)
        
        # Pre-select some common categories
        for cat in ["Clothing", "Electronics", "Fitness"]:
            if cat in self.category_vars:
                self.category_vars[cat].set(True)
        
        # Target demographics
        demo_frame = ttk.Frame(form_frame, style="TFrame")
        demo_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(demo_frame, text="Target Demographic:").pack(side=tk.LEFT, padx=(0, 10))
        
        demographics = [
            "Any", "Young Adults", "Professionals", "Parents", "Seniors", "Teenagers",
            "Children", "Men", "Women", "Families", "Students"
        ]
        
        self.demographic_var = tk.StringVar(value=demographics[0])
        demo_dropdown = ttk.Combobox(demo_frame, textvariable=self.demographic_var, 
                                    values=demographics, width=15, state="readonly")
        demo_dropdown.pack(side=tk.LEFT)
        
        # Time horizon
        time_frame = ttk.Frame(form_frame, style="TFrame")
        time_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(time_frame, text="Prediction Timeframe:").pack(side=tk.LEFT, padx=(0, 10))
        
        timeframes = ["4 weeks", "8 weeks", "12 weeks", "24 weeks"]
        self.timeframe_var = tk.StringVar(value=timeframes[1])
        time_dropdown = ttk.Combobox(time_frame, textvariable=self.timeframe_var, 
                                    values=timeframes, width=10, state="readonly")
        time_dropdown.pack(side=tk.LEFT)
    
    def search_products(self):
        """Execute product search based on input criteria"""
        if not self.initialized:
            messagebox.showerror("Error", "Product trend predictor not initialized.")
            return
        
        try:
            # Clear previous results
            for widget in self.results_frame.winfo_children():
                widget.destroy()
                
            # Clear previous visualization
            for widget in self.viz_container.winfo_children():
                widget.destroy()
            
            # Update status
            self.status_var.set("Searching for trending products...")
            self.root.update_idletasks()
            
            # Get input values
            try:
                budget_min = float(self.min_price_var.get())
                budget_max = float(self.max_price_var.get())
            except ValueError:
                messagebox.showerror("Input Error", "Price range must be numeric values.")
                return
            
            # Get selected categories
            selected_categories = [cat for cat, var in self.category_vars.items() if var.get()]
            
            # Get demographic
            demographic = self.demographic_var.get()
            if demographic == "Any":
                demographic = None
                
            # Get timeframe in weeks
            timeframe_text = self.timeframe_var.get()
            trend_window_weeks = int(timeframe_text.split()[0])
            
            # Prepare client inputs
            client_inputs = {
                'budget_min': budget_min,
                'budget_max': budget_max,
                'categories': selected_categories,
                'target_demographic': demographic,
                'trend_window_weeks': trend_window_weeks
            }
            
            # Get recommendations
            recommendations = self.predictor.get_product_recommendations(client_inputs)
            
            # Display results
            if recommendations['status'] == 'success':
                self.display_results(recommendations)
            else:
                # No results found
                no_results = ttk.Label(self.results_frame, 
                                     text=recommendations['message'],
                                     style="Subheader.TLabel")
                no_results.pack(pady=30)
                self.status_var.set("No matching products found.")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.status_var.set("Error occurred during search.")
    
    def display_results(self, recommendations):
        """Display the product recommendations"""
        context = recommendations['context']
        products = recommendations['products']
        
        # Clear existing content
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        # Create notebook for tabbed results
        self.results_notebook = ttk.Notebook(self.results_frame)
        self.results_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tab frames
        overview_tab = ttk.Frame(self.results_notebook)
        category_tab = ttk.Frame(self.results_notebook)
        details_tab = ttk.Frame(self.results_notebook)
        
        self.results_notebook.add(overview_tab, text="Overview")
        self.results_notebook.add(category_tab, text="By Category")
        self.results_notebook.add(details_tab, text="Detailed Results")
        
        # ----- OVERVIEW TAB -----
        
        # Create header with summary info
        header_frame = ttk.Frame(overview_tab, style="TFrame")
        header_frame.pack(fill=tk.X, pady=(10, 15))
        
        # Summary text
        summary_text = (f"Predicted trending products for {context['prediction_date']} "
                      f"({context['weeks_ahead']} weeks ahead)")
        summary = ttk.Label(header_frame, text=summary_text, style="Header.TLabel")
        summary.pack(anchor=tk.W)
        
        # Additional context
        context_text = (f"Top category: {context['top_category']} | "
                      f"Average growth: {context['avg_growth']}% | "
                      f"Price range: ${context['price_range'][0]:.2f} - ${context['price_range'][1]:.2f}")
        context_label = ttk.Label(header_frame, text=context_text, foreground=self.light_text)
        context_label.pack(anchor=tk.W)
        
        # Create visualization in the overview tab
        self.create_overview_visualization(overview_tab, products, context)
        
        # ----- CATEGORY TAB -----
        
        # Group products by category
        categories = {}
        for product in products:
            category = product['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(product)
        
        # Create category summary
        cat_header = ttk.Frame(category_tab, style="TFrame")
        cat_header.pack(fill=tk.X, pady=10)
        
        cat_title = ttk.Label(cat_header, text="Products by Category", style="Header.TLabel")
        cat_title.pack(anchor=tk.W)
        
        # Create inner notebook for categories
        cat_notebook = ttk.Notebook(category_tab)
        cat_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add a tab for each category
        for category, cat_products in categories.items():
            cat_tab = ttk.Frame(cat_notebook)
            cat_notebook.add(cat_tab, text=category)
            
            # Add header with category info
            cat_info = ttk.Frame(cat_tab, style="TFrame")
            cat_info.pack(fill=tk.X, pady=10)
            
            # Category stats
            avg_trend = sum(p['predicted_trend'] for p in cat_products) / len(cat_products)
            avg_growth = sum(p['growth_percentage'] for p in cat_products) / len(cat_products)
            
            ttk.Label(cat_info, text=f"{category} - {len(cat_products)} products", 
                    style="Subheader.TLabel").pack(anchor=tk.W)
            ttk.Label(cat_info, text=f"Average trend score: {avg_trend:.1f}/100").pack(anchor=tk.W)
            ttk.Label(cat_info, text=f"Average growth: {avg_growth:.1f}%").pack(anchor=tk.W)
            
            # List products in this category
            products_frame = ttk.Frame(cat_tab, style="TFrame")
            products_frame.pack(fill=tk.BOTH, expand=True, pady=10)
            
            # Sort products by trend score
            cat_products.sort(key=lambda x: x['predicted_trend'], reverse=True)
            
            # Display compact product cards
            for i, product in enumerate(cat_products, 1):
                self.create_compact_product_card(products_frame, product, i)
        
        # ----- DETAILS TAB -----
        
        # Create scrollable container for detailed product cards
        details_canvas = tk.Canvas(details_tab, bg=self.bg_color, highlightthickness=0)
        details_scrollbar = ttk.Scrollbar(details_tab, orient="vertical", command=details_canvas.yview)
        
        # Configure canvas
        details_canvas.configure(yscrollcommand=details_scrollbar.set)
        details_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        details_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Frame inside canvas for results
        details_container = ttk.Frame(details_canvas, style="TFrame")
        details_canvas.create_window((0, 0), window=details_container, anchor="nw")
        
        # Sort products by trend score
        sorted_products = sorted(products, key=lambda p: p['predicted_trend'], reverse=True)
        
        # Display each product as a detailed card
        for i, product in enumerate(sorted_products, 1):
            self.create_product_card(details_container, product, i)
        
        # Update canvas scroll region
        details_container.update_idletasks()
        details_canvas.configure(scrollregion=details_canvas.bbox("all"))
        details_canvas.config(width=details_tab.winfo_width())
        
        # Update status
        self.status_var.set(f"Found {len(products)} trending products across {len(categories)} categories.")
    
    def create_overview_visualization(self, parent, products, context):
        """Create visualization of products in the overview tab"""
        # Top products section
        top_frame = ttk.LabelFrame(parent, text="Top 5 Trending Products", padding=10)
        top_frame.pack(fill=tk.X, pady=10)
        
        # Create horizontal figure for overview
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Sort products by trend score
        sorted_products = sorted(products, key=lambda p: p['predicted_trend'], reverse=True)[:5]
        
        # Extract data for visualization
        names = [self.shorten_name(p['product_name'], max_length=25) for p in sorted_products]
        trend_scores = [p['predicted_trend'] for p in sorted_products]
        growth_rates = [p['growth_percentage'] for p in sorted_products]
        categories = [p['category'] for p in sorted_products]
        
        # Reverse lists for better visualization (highest at top)
        names.reverse()
        trend_scores.reverse()
        growth_rates.reverse()
        categories.reverse()
        
        # Create colormap based on categories
        unique_categories = list(set(categories))
        colors = plt.cm.tab10(range(len(unique_categories)))
        category_colors = {cat: colors[i] for i, cat in enumerate(unique_categories)}
        bar_colors = [category_colors[cat] for cat in categories]
        
        # Create horizontal bar chart
        bars = ax.barh(names, trend_scores, height=0.7, color=bar_colors)
        
        # Add growth rate and category annotations
        for i, bar in enumerate(bars):
            width = bar.get_width()
            y_pos = bar.get_y() + bar.get_height()/2
            
            # Add growth annotation
            ax.text(width + 2, y_pos, f'↑ {growth_rates[i]:.1f}%', 
                   va='center', fontsize=9, color=self.get_trend_color(trend_scores[i]))
            
            # Add category annotation at the beginning of the bar
            ax.text(0, y_pos, categories[i], va='center', ha='left', 
                   fontsize=8, color='white', fontweight='bold',
                   bbox=dict(facecolor=category_colors[categories[i]], alpha=0.8, pad=2))
        
        # Style the plot
        ax.set_xlabel('Trend Score')
        ax.set_title('Top Trending Products')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        ax.set_xlim(0, max(trend_scores) * 1.3)  # Give space for annotations
        
        plt.tight_layout()
        
        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=top_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create category distribution visualization
        cat_frame = ttk.LabelFrame(parent, text="Category Distribution", padding=10)
        cat_frame.pack(fill=tk.X, pady=10)
        
        # Create pie chart for category distribution
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        
        # Count products per category
        cat_counts = {}
        for p in products:
            cat = p['category']
            if cat in cat_counts:
                cat_counts[cat] += 1
            else:
                cat_counts[cat] = 1
        
        # Create pie chart
        labels = list(cat_counts.keys())
        sizes = list(cat_counts.values())
        explode = [0.1 if cat == context['top_category'] else 0 for cat in labels]
        
        ax2.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
               shadow=True, startangle=90, colors=plt.cm.tab10.colors[:len(labels)])
        ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        plt.title("Distribution by Category")
        plt.tight_layout()
        
        # Embed in Tkinter
        canvas2 = FigureCanvasTkAgg(fig2, master=cat_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_compact_product_card(self, parent, product, rank):
        """Create a compact card for category view"""
        # Card container
        card = ttk.Frame(parent, style="Card.TFrame")
        card.pack(fill=tk.X, pady=5, padx=5, ipadx=5, ipady=5)
        
        # Product info
        info_frame = ttk.Frame(card, style="Card.TFrame")
        info_frame.pack(fill=tk.X, padx=5, pady=2)
        
        # Rank and name
        rank_text = f"#{rank}"
        rank_label = ttk.Label(info_frame, text=rank_text, 
                             font=("Segoe UI", 12, "bold"), foreground=self.primary_color)
        rank_label.pack(side=tk.LEFT, padx=(0, 5))
        
        name_label = ttk.Label(info_frame, text=product['product_name'], 
                             font=("Segoe UI", 12))
        name_label.pack(side=tk.LEFT)
        
        # Trend score and growth on right
        metrics_frame = ttk.Frame(info_frame, style="Card.TFrame")
        metrics_frame.pack(side=tk.RIGHT)
        
        trend_score = product['predicted_trend']
        trend_color = self.get_trend_color(trend_score)
        
        # Create a small trend indicator bar
        indicator_canvas = tk.Canvas(metrics_frame, width=50, height=15, 
                                    bg="white", highlightthickness=0)
        indicator_canvas.pack(side=tk.RIGHT, padx=(5, 0))
        
        # Draw trend indicator bar
        bar_width = int((trend_score / 100) * 50)
        indicator_canvas.create_rectangle(0, 0, bar_width, 15, 
                                        fill=trend_color, outline="")
        
        # Add trend score text
        score_text = f"{trend_score:.0f}"
        trend_label = ttk.Label(metrics_frame, text=score_text, 
                              foreground=trend_color, font=("Segoe UI", 10, "bold"))
        trend_label.pack(side=tk.RIGHT)
        
        # Add growth percentage
        growth = product['growth_percentage']
        growth_color = self.success_color if growth > 0 else self.warning_color
        growth_text = f"↑{growth:.1f}%"
        
        growth_label = ttk.Label(metrics_frame, text=growth_text, 
                               foreground=growth_color, font=("Segoe UI", 10))
        growth_label.pack(side=tk.RIGHT, padx=10)
        
        # Add price
        price_label = ttk.Label(metrics_frame, text=f"${product['price']:.2f}", 
                              font=("Segoe UI", 10))
        price_label.pack(side=tk.RIGHT, padx=10)
    
    def create_product_card(self, parent, product, rank):
        """Create a card for displaying a product"""
        # Card container with white background
        card = ttk.Frame(parent, style="Card.TFrame")
        card.pack(fill=tk.X, pady=10, padx=5, ipadx=10, ipady=10)
        
        # Product header with rank
        header_frame = ttk.Frame(card, style="Card.TFrame")
        header_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Rank and name
        rank_text = f"#{rank}"
        rank_label = ttk.Label(header_frame, text=rank_text, 
                             font=("Segoe UI", 14, "bold"), foreground=self.primary_color)
        rank_label.pack(side=tk.LEFT, padx=(0, 10))
        
        name_label = ttk.Label(header_frame, text=product['product_name'], 
                             font=("Segoe UI", 14, "bold"))
        name_label.pack(side=tk.LEFT)
        
        # Trend score on right
        trend_score = product['predicted_trend']
        trend_color = self.get_trend_color(trend_score)
        trend_text = f"Trend Score: {trend_score}/100"
        
        trend_label = ttk.Label(header_frame, text=trend_text, 
                              font=("Segoe UI", 12, "bold"), foreground=trend_color)
        trend_label.pack(side=tk.RIGHT)
        
        # Details section
        details_frame = ttk.Frame(card, style="Card.TFrame")
        details_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Create two columns
        left_col = ttk.Frame(details_frame, style="Card.TFrame")
        left_col.pack(side=tk.LEFT, fill=tk.Y, expand=True)
        
        right_col = ttk.Frame(details_frame, style="Card.TFrame")
        right_col.pack(side=tk.RIGHT, fill=tk.Y, expand=True)
        
        # Left column details
        ttk.Label(left_col, text=f"Category: {product['category']}").pack(anchor=tk.W, pady=2)
        ttk.Label(left_col, text=f"Price: ${product['price']:.2f}").pack(anchor=tk.W, pady=2)
        ttk.Label(left_col, text=f"Target: {product['target_demographic']}").pack(anchor=tk.W, pady=2)
        
        # Right column details - growth and metrics
        growth = product['growth_percentage']
        growth_color = self.success_color if growth > 0 else self.warning_color
        growth_text = f"Growth: {growth:.1f}% in {product['trend_window_weeks']} weeks"
        
        ttk.Label(right_col, text=growth_text, foreground=growth_color).pack(anchor=tk.W, pady=2)
        
        if 'social_mentions_weekly' in product:
            ttk.Label(right_col, text=f"Social Mentions: {product['social_mentions_weekly']} weekly").pack(anchor=tk.W, pady=2)
        
        if 'search_volume_weekly' in product:
            ttk.Label(right_col, text=f"Search Volume: {product['search_volume_weekly']} weekly").pack(anchor=tk.W, pady=2)
            
        if 'seasonality' in product:
            seasonality = product['seasonality']
            seasonality_text = "Low" if seasonality <= 2 else "Medium" if seasonality <= 3 else "High"
            ttk.Label(right_col, text=f"Seasonality: {seasonality_text}").pack(anchor=tk.W, pady=2)

    def create_trend_visualization(self, products):
        """Create visualization of product trends"""
        # Clear previous widgets
        for widget in self.viz_container.winfo_children():
            widget.destroy()
            
        # Create figure
        fig, ax = plt.subplots(figsize=(5, 7))  # Taller figure for better visibility
        
        # Extract data for visualization
        names = [self.shorten_name(p['product_name'], max_length=25) for p in products]
        trend_scores = [p['predicted_trend'] for p in products]
        growth_rates = [p['growth_percentage'] for p in products]
        
        # Reverse the lists to display highest trending at the top
        names.reverse()
        trend_scores.reverse()
        growth_rates.reverse()
        
        # Create horizontal bar chart
        bars = ax.barh(names, trend_scores, height=0.6)
        
        # Set colors based on trend score
        for i, bar in enumerate(bars):
            bar.set_color(self.get_trend_color(trend_scores[i], as_hex=True))
            
            # Add growth rate annotations
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                   f'↑ {growth_rates[i]:.1f}%', va='center')
        
        # Style the plot
        ax.set_xlabel('Trend Score')
        ax.set_title('Predicted Product Trends')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        ax.set_xlim(0, max(trend_scores) * 1.2)  # Give space for annotations
        
        plt.tight_layout()
        
        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.viz_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def shorten_name(self, name, max_length=30):
        """Shorten product name for display"""
        if len(name) <= max_length:
            return name
        return name[:max_length-3] + "..."
    
    def get_trend_color(self, trend_score, as_hex=False):
        """Get color based on trend score"""
        if trend_score >= 75:
            color = "#4CAF50" if as_hex else self.success_color  # Green
        elif trend_score >= 50:
            color = "#8BC34A" if as_hex else self.success_color  # Light Green
        elif trend_score >= 25:
            color = "#FFC107" if as_hex else self.warning_color  # Amber
        else:
            color = "#FF5722" if as_hex else self.secondary_color  # Deep Orange
        
        return color