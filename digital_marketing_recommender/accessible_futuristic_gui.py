import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import os
import sys
from simple_recommender import SimpleMarketingRecommender

class FuturisticTheme:
    """Futuristic theme colors and styles"""
    BG_DARK = "#1a1a2e"  # Dark blue background
    BG_MEDIUM = "#16213e"  # Medium blue background
    BG_LIGHT = "#0f3460"  # Light blue background
    ACCENT = "#e94560"  # Neon red/pink accent
    TEXT_PRIMARY = "#ffffff"  # White text
    TEXT_SECONDARY = "#b0b0b0"  # Light gray text
    SUCCESS = "#4ecca3"  # Green for positive values
    
    FONT_HEADER = ("Segoe UI", 16, "bold")
    FONT_SUBHEADER = ("Segoe UI", 11)
    FONT_LABEL = ("Segoe UI", 9)
    FONT_RESULTS = ("Consolas", 9)

class ToolTip:
    """Tooltip class for adding hover information to widgets"""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)
    
    def show_tooltip(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        
        # Create tooltip window
        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")
        
        # Create tooltip content
        frame = tk.Frame(self.tooltip_window, 
                        background=FuturisticTheme.BG_DARK, 
                        borderwidth=1, 
                        relief="solid")
        frame.pack(fill="both", expand=True)
        
        label = tk.Label(frame, 
                        text=self.text, 
                        background=FuturisticTheme.BG_DARK,
                        foreground=FuturisticTheme.TEXT_PRIMARY,
                        font=("Segoe UI", 9),
                        justify="left",
                        wraplength=250,
                        padx=5, pady=5)
        label.pack()
    
    def hide_tooltip(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None

class AccessibleFuturisticGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Digital Marketing Navigator")
        self.root.geometry("1000x700")  # More reasonable default size
        self.root.minsize(800, 600)  # Set minimum window size
        self.root.configure(bg=FuturisticTheme.BG_DARK)
        
        # Initialize recommender
        self.recommender = SimpleMarketingRecommender()
        if not self.recommender.strategies:
            messagebox.showerror("Error", "Failed to load marketing strategies data.")
            self.root.destroy()
            return
            
        # Extract industries
        self.all_industries = self.extract_all_industries()
        
        # Create UI elements
        self.setup_styles()
        self.create_widgets()
        
    def extract_all_industries(self):
        """Extract all unique industries from the dataset"""
        all_industries = set()
        for strategy in self.recommender.strategies:
            industries = strategy['best_for_industry'].replace('"', '').split(',')
            all_industries.update(industries)
        return sorted(list(all_industries))
    
    def setup_styles(self):
        """Configure ttk styles with futuristic theme"""
        style = ttk.Style()
        
        # Configure base styles
        style.configure("TFrame", background=FuturisticTheme.BG_DARK)
        style.configure("TLabel", background=FuturisticTheme.BG_DARK, foreground=FuturisticTheme.TEXT_PRIMARY, font=FuturisticTheme.FONT_LABEL)
        
        # Buttons
        style.configure("TButton", 
                       background=FuturisticTheme.ACCENT,
                       foreground=FuturisticTheme.TEXT_PRIMARY,
                       font=FuturisticTheme.FONT_LABEL)
        
        style.configure("Accent.TButton", 
                       background=FuturisticTheme.ACCENT,
                       foreground=FuturisticTheme.TEXT_PRIMARY,
                       font=("Segoe UI", 10, "bold"))
        
        style.configure("Help.TButton", 
                       background=FuturisticTheme.BG_MEDIUM,
                       foreground=FuturisticTheme.TEXT_PRIMARY,
                       font=("Segoe UI", 10, "bold"))
        
        # Combobox
        style.configure("TCombobox",
                       fieldbackground=FuturisticTheme.BG_MEDIUM,
                       background=FuturisticTheme.BG_MEDIUM,
                       foreground=FuturisticTheme.TEXT_PRIMARY)
        
        # Radiobuttons
        style.configure("TRadiobutton",
                       background=FuturisticTheme.BG_MEDIUM,
                       foreground=FuturisticTheme.TEXT_PRIMARY)
    
    def create_widgets(self):
        # Create main container with padding
        main_container = tk.Frame(self.root, bg=FuturisticTheme.BG_DARK)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header with help button
        self.header_frame = tk.Frame(main_container, bg=FuturisticTheme.BG_DARK, pady=5)
        self.header_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Title
        title_label = tk.Label(
            self.header_frame, 
            text="DIGITAL MARKETING NAVIGATOR",
            font=FuturisticTheme.FONT_HEADER,
            fg=FuturisticTheme.ACCENT,
            bg=FuturisticTheme.BG_DARK
        )
        title_label.pack(side=tk.LEFT)
        
        # Help button
        help_button = ttk.Button(
            self.header_frame, 
            text="?", 
            style="Help.TButton",
            width=3,
            command=self.show_help
        )
        help_button.pack(side=tk.RIGHT, padx=10)
        ToolTip(help_button, "Click for help and instructions")
        
        # Create a tabbed interface for more accessible layout
        notebook = ttk.Notebook(main_container)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Input tab - Profile and settings
        input_frame = tk.Frame(notebook, bg=FuturisticTheme.BG_MEDIUM, padx=15, pady=15)
        
        # Results tab
        results_frame = tk.Frame(notebook, bg=FuturisticTheme.BG_MEDIUM, padx=15, pady=15)
        
        # Add tabs to notebook
        notebook.add(input_frame, text="Business Profile")
        notebook.add(results_frame, text="Recommendations")
        
        # Bind tab change to update display
        notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)
        self.notebook = notebook
        
        # ===== INPUT TAB =====
        # Instructions at the top
        instruction_text = "Complete your business profile below and click 'Generate Recommendations'"
        instruction_label = tk.Label(
            input_frame,
            text=instruction_text,
            font=FuturisticTheme.FONT_SUBHEADER,
            fg=FuturisticTheme.TEXT_SECONDARY,
            bg=FuturisticTheme.BG_MEDIUM
        )
        instruction_label.pack(anchor="w", pady=(0, 15))
        
        # Input form - Using Grid for better organization
        form_frame = tk.Frame(input_frame, bg=FuturisticTheme.BG_MEDIUM)
        form_frame.pack(fill=tk.BOTH, expand=True)
        
        # Industry selection - row 0
        industry_label = tk.Label(
            form_frame,
            text="INDUSTRY",
            font=("Segoe UI", 10, "bold"),
            fg=FuturisticTheme.ACCENT,
            bg=FuturisticTheme.BG_MEDIUM
        )
        industry_label.grid(row=0, column=0, sticky="w", pady=(10, 5))
        
        self.industry_var = tk.StringVar()
        industry_combo = ttk.Combobox(
            form_frame, 
            textvariable=self.industry_var,
            width=30,
            state="readonly"
        )
        industry_combo['values'] = self.all_industries
        industry_combo.grid(row=0, column=1, sticky="w", padx=10, pady=(10, 5))
        industry_combo.current(0) if self.all_industries else None
        ToolTip(industry_combo, "Select your business industry. This helps filter strategies that work best for your sector.")
        
        # Budget level - row 1
        budget_label = tk.Label(
            form_frame,
            text="BUDGET LEVEL",
            font=("Segoe UI", 10, "bold"),
            fg=FuturisticTheme.ACCENT,
            bg=FuturisticTheme.BG_MEDIUM
        )
        budget_label.grid(row=1, column=0, sticky="w", pady=(15, 5))
        
        budget_frame = tk.Frame(form_frame, bg=FuturisticTheme.BG_MEDIUM)
        budget_frame.grid(row=1, column=1, sticky="w", padx=10, pady=(15, 5))
        
        self.budget_var = tk.IntVar(value=3)
        
        for i in range(1, 6):
            rb = ttk.Radiobutton(
                budget_frame, 
                text=str(i), 
                variable=self.budget_var, 
                value=i
            )
            rb.pack(side=tk.LEFT, padx=10)
            
        budget_desc = tk.Label(
            form_frame,
            text="(1=Very Low, 5=Very High)",
            font=FuturisticTheme.FONT_LABEL,
            fg=FuturisticTheme.TEXT_SECONDARY,
            bg=FuturisticTheme.BG_MEDIUM
        )
        budget_desc.grid(row=1, column=2, sticky="w", pady=(15, 5))
        ToolTip(budget_frame, "Rate your available marketing budget. Lower values mean strategies that require less investment.")
        
        # Technical skills - row 2
        tech_label = tk.Label(
            form_frame,
            text="TECHNICAL SKILLS",
            font=("Segoe UI", 10, "bold"),
            fg=FuturisticTheme.ACCENT,
            bg=FuturisticTheme.BG_MEDIUM
        )
        tech_label.grid(row=2, column=0, sticky="w", pady=(15, 5))
        
        tech_frame = tk.Frame(form_frame, bg=FuturisticTheme.BG_MEDIUM)
        tech_frame.grid(row=2, column=1, sticky="w", padx=10, pady=(15, 5))
        
        self.tech_var = tk.IntVar(value=3)
        
        for i in range(1, 6):
            rb = ttk.Radiobutton(
                tech_frame, 
                text=str(i), 
                variable=self.tech_var, 
                value=i
            )
            rb.pack(side=tk.LEFT, padx=10)
            
        tech_desc = tk.Label(
            form_frame,
            text="(1=Beginner, 5=Expert)",
            font=FuturisticTheme.FONT_LABEL,
            fg=FuturisticTheme.TEXT_SECONDARY,
            bg=FuturisticTheme.BG_MEDIUM
        )
        tech_desc.grid(row=2, column=2, sticky="w", pady=(15, 5))
        ToolTip(tech_frame, "Rate your team's technical expertise. Lower values suggest strategies requiring less technical knowledge.")
        
        # Time available - row 3
        time_label = tk.Label(
            form_frame,
            text="TIME AVAILABLE",
            font=("Segoe UI", 10, "bold"),
            fg=FuturisticTheme.ACCENT,
            bg=FuturisticTheme.BG_MEDIUM
        )
        time_label.grid(row=3, column=0, sticky="w", pady=(15, 5))
        
        time_frame = tk.Frame(form_frame, bg=FuturisticTheme.BG_MEDIUM)
        time_frame.grid(row=3, column=1, sticky="w", padx=10, pady=(15, 5))
        
        self.time_var = tk.IntVar(value=3)
        
        for i in range(1, 6):
            rb = ttk.Radiobutton(
                time_frame, 
                text=str(i), 
                variable=self.time_var, 
                value=i
            )
            rb.pack(side=tk.LEFT, padx=10)
            
        time_desc = tk.Label(
            form_frame,
            text="(1=Very Limited, 5=Abundant)",
            font=FuturisticTheme.FONT_LABEL,
            fg=FuturisticTheme.TEXT_SECONDARY,
            bg=FuturisticTheme.BG_MEDIUM
        )
        time_desc.grid(row=3, column=2, sticky="w", pady=(15, 5))
        ToolTip(time_frame, "Rate the time you can dedicate to marketing. Lower values mean strategies requiring less time investment.")
        
        # Audience size - row 4
        audience_label = tk.Label(
            form_frame,
            text="TARGET AUDIENCE SIZE",
            font=("Segoe UI", 10, "bold"),
            fg=FuturisticTheme.ACCENT,
            bg=FuturisticTheme.BG_MEDIUM
        )
        audience_label.grid(row=4, column=0, sticky="w", pady=(15, 5))
        
        audience_frame = tk.Frame(form_frame, bg=FuturisticTheme.BG_MEDIUM)
        audience_frame.grid(row=4, column=1, sticky="w", padx=10, pady=(15, 5))
        
        self.audience_var = tk.IntVar(value=3)
        
        for i in range(1, 6):
            rb = ttk.Radiobutton(
                audience_frame, 
                text=str(i), 
                variable=self.audience_var, 
                value=i
            )
            rb.pack(side=tk.LEFT, padx=10)
            
        audience_desc = tk.Label(
            form_frame,
            text="(1=Very Small, 5=Very Large)",
            font=FuturisticTheme.FONT_LABEL,
            fg=FuturisticTheme.TEXT_SECONDARY,
            bg=FuturisticTheme.BG_MEDIUM
        )
        audience_desc.grid(row=4, column=2, sticky="w", pady=(15, 5))
        ToolTip(audience_frame, "Rate the size of your target audience. Higher values mean strategies that work well for larger markets.")
        
        # Marketing Goal - row 5
        goal_label = tk.Label(
            form_frame,
            text="PRIMARY MARKETING GOAL",
            font=("Segoe UI", 10, "bold"),
            fg=FuturisticTheme.ACCENT,
            bg=FuturisticTheme.BG_MEDIUM
        )
        goal_label.grid(row=5, column=0, sticky="w", pady=(15, 5))
        
        goal_frame = tk.Frame(form_frame, bg=FuturisticTheme.BG_MEDIUM)
        goal_frame.grid(row=5, column=1, columnspan=2, sticky="w", padx=10, pady=(15, 5))
        
        self.goal_var = tk.StringVar(value="awareness")
        
        goals = [
            ("Increase conversion rates", "conversion", "Increase conversion rates"),
            ("Boost brand awareness", "awareness", "Boost brand awareness"),
            ("Generate more leads", "leads", "Generate more leads"),
            ("Improve customer retention", "retention", "Improve customer retention")
        ]
        
        # Create goal options in a 2x2 grid for better spacing
        for i, (text, value, tooltip) in enumerate(goals):
            row, col = divmod(i, 2)
            goal_option = ttk.Radiobutton(
                goal_frame,
                text=text,
                variable=self.goal_var,
                value=value
            )
            goal_option.grid(row=row, column=col, sticky="w", padx=10, pady=5)
            ToolTip(goal_option, tooltip)
        
        # Generate button - row 6
        button_frame = tk.Frame(form_frame, bg=FuturisticTheme.BG_MEDIUM)
        button_frame.grid(row=6, column=0, columnspan=3, pady=25)
        
        generate_button = ttk.Button(
            button_frame,
            text="GENERATE RECOMMENDATIONS",
            command=self.generate_recommendations,
            style="Accent.TButton"
        )
        generate_button.pack(ipadx=20, ipady=8)
        ToolTip(generate_button, "Click to analyze your business profile and generate personalized marketing strategies")
        
        # ===== RESULTS TAB =====
        results_container = tk.Frame(results_frame, bg=FuturisticTheme.BG_MEDIUM)
        results_container.pack(fill=tk.BOTH, expand=True)
        
        # Results display
        self.results_text = scrolledtext.ScrolledText(
            results_container,
            wrap=tk.WORD,
            font=FuturisticTheme.FONT_RESULTS,
            bg=FuturisticTheme.BG_MEDIUM,
            fg=FuturisticTheme.TEXT_PRIMARY,
            bd=0,
            height=25
        )
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.results_text.insert(tk.END, "Complete your business profile and click GENERATE RECOMMENDATIONS to get personalized marketing strategies.")
        self.results_text.configure(state="disabled")
        
        # Status bar at the bottom
        self.status_var = tk.StringVar(value="Ready")
        status_bar = tk.Label(
            main_container,
            textvariable=self.status_var,
            bg=FuturisticTheme.BG_LIGHT,
            fg=FuturisticTheme.TEXT_SECONDARY,
            anchor=tk.W,
            padx=10,
            pady=5
        )
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
    
    def on_tab_change(self, event):
        """Handle tab change events"""
        selected_tab = self.notebook.index(self.notebook.select())
        if selected_tab == 1:  # Results tab
            # You could refresh the results display here if needed
            pass
    
    def show_help(self):
        """Show help dialog with usage instructions"""
        help_window = tk.Toplevel(self.root)
        help_window.title("How to Use the Digital Marketing Navigator")
        help_window.geometry("600x500")
        help_window.configure(bg=FuturisticTheme.BG_DARK)
        
        # Make it modal (user needs to close it before continuing)
        help_window.transient(self.root)
        help_window.grab_set()
        
        # Main frame
        main_frame = tk.Frame(help_window, bg=FuturisticTheme.BG_DARK, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title = tk.Label(
            main_frame,
            text="HOW TO USE THIS TOOL",
            font=("Segoe UI", 16, "bold"),
            fg=FuturisticTheme.ACCENT,
            bg=FuturisticTheme.BG_DARK
        )
        title.pack(pady=(0, 20))
        
        # Help content in a scrollable frame
        help_frame = tk.Frame(main_frame, bg=FuturisticTheme.BG_MEDIUM, padx=15, pady=15)
        help_frame.pack(fill=tk.BOTH, expand=True)
        
        # Scrollable text widget for help content
        help_text = scrolledtext.ScrolledText(
            help_frame,
            bg=FuturisticTheme.BG_MEDIUM,
            fg=FuturisticTheme.TEXT_PRIMARY,
            font=("Segoe UI", 10),
            wrap=tk.WORD,
            padx=10,
            pady=10
        )
        help_text.pack(fill=tk.BOTH, expand=True)
        
        # Insert help content
        help_content = """
QUICK START GUIDE:

1. SELECT YOUR INDUSTRY from the dropdown menu

2. SET YOUR BUSINESS PROFILE using the rating scales:
   • Budget Level: Available marketing funds
   • Technical Skills: Your team's expertise
   • Time Available: Hours you can dedicate
   • Target Audience Size: Market reach

3. CHOOSE PRIMARY GOAL:
   • CONVERSION: Turn visitors into customers
   • AWARENESS: Increase brand visibility
   • LEADS: Generate potential customers
   • RETENTION: Keep existing customers

4. CLICK "GENERATE RECOMMENDATIONS"

5. REVIEW RESULTS in the Recommendations tab:
   • Match Score shows compatibility
   • Resource Requirements show what you need
   • Expected Outcomes show likely results

UNDERSTANDING THE VISUALS:

■■■□□ (3/5) = Medium level (3 out of 5)
■■■■■ (5/5) = Maximum level (5 out of 5)

TIP: Focus on strategies with high match scores that
align with your available resources and goals.
"""
        
        help_text.insert(tk.END, help_content)
        help_text.configure(state="disabled")  # Make it read-only
        
        # Close button
        close_button = ttk.Button(
            main_frame,
            text="CLOSE",
            command=help_window.destroy,
            style="Accent.TButton"
        )
        close_button.pack(pady=(15, 0))
    
    def generate_recommendations(self):
        """Generate and display recommendations based on user inputs"""
        try:
            # Switch to results tab
            self.notebook.select(1)
            
            # Clear previous results
            self.results_text.configure(state="normal")
            self.results_text.delete(1.0, tk.END)
            
            # Update status
            self.status_var.set("Analyzing data and generating recommendations...")
            self.root.update_idletasks()
            
            # Get user inputs
            industry = self.industry_var.get()
            budget_level = self.budget_var.get()
            technical_skill = self.tech_var.get()
            time_available = self.time_var.get()
            audience_size = self.audience_var.get()
            goal = self.goal_var.get()
            
            # Validate industry selection
            if not industry:
                messagebox.showwarning("Input Error", "Please select an industry.")
                self.status_var.set("Ready")
                return
            
            # Create user preferences dictionary
            user_prefs = {
                'budget_level': budget_level,
                'technical_skill': technical_skill,
                'time_available': time_available,
                'goal': goal,
                'industry': industry,
                'audience_size': audience_size
            }
            
            # Get recommendations
            recommendations = self.recommender.get_recommendations(user_prefs, top_n=3)
            
            if not recommendations:
                self.results_text.insert(tk.END, "No matching strategies found. Try adjusting your criteria.")
                self.status_var.set("No matches found")
                return
            
            # Display recommendations with futuristic styling
            self.results_text.insert(tk.END, "⋆⋆⋆ TOP RECOMMENDED STRATEGIES ⋆⋆⋆\n\n", "header")
            
            # Add tags for styling
            self.results_text.tag_configure("header", foreground=FuturisticTheme.ACCENT, font=("Consolas", 12, "bold"))
            self.results_text.tag_configure("title", foreground=FuturisticTheme.ACCENT, font=("Consolas", 11, "bold"))
            self.results_text.tag_configure("subtitle", foreground=FuturisticTheme.TEXT_PRIMARY, font=("Consolas", 9, "bold"))
            self.results_text.tag_configure("info", foreground=FuturisticTheme.TEXT_SECONDARY)
            self.results_text.tag_configure("highlight", foreground=FuturisticTheme.ACCENT)
            self.results_text.tag_configure("match", foreground=FuturisticTheme.SUCCESS, font=("Consolas", 9, "bold"))
            
            for i, (strategy, similarity) in enumerate(recommendations, 1):
                # Strategy header
                self.results_text.insert(tk.END, f"STRATEGY {i}: ", "title")
                self.results_text.insert(tk.END, f"{strategy['strategy_name']}\n", "highlight")
                self.results_text.insert(tk.END, f"MATCH SCORE: {similarity:.2f}\n", "match")
                self.results_text.insert(tk.END, f"Best for: {strategy['best_for_industry']}\n\n", "info")
                
                # Resource Requirements
                self.results_text.insert(tk.END, "RESOURCE REQUIREMENTS:\n", "subtitle")
                
                # Create custom visual indicators with symbols
                budget = "■" * strategy['budget_required'] + "□" * (5 - strategy['budget_required'])
                tech = "■" * strategy['technical_expertise'] + "□" * (5 - strategy['technical_expertise'])
                time = "■" * strategy['time_investment'] + "□" * (5 - strategy['time_investment'])
                
                self.results_text.insert(tk.END, f"  Budget required:    {budget} ({strategy['budget_required']}/5)\n", "info")
                self.results_text.insert(tk.END, f"  Technical expertise: {tech} ({strategy['technical_expertise']}/5)\n", "info")
                self.results_text.insert(tk.END, f"  Time investment:    {time} ({strategy['time_investment']}/5)\n\n", "info")
                
                # Expected Outcomes
                self.results_text.insert(tk.END, "EXPECTED OUTCOMES:\n", "subtitle")
                
                # Create custom visual indicators for outcomes
                conv = "■" * strategy['conversion_rate'] + "□" * (5 - strategy['conversion_rate'])
                aware = "■" * strategy['brand_awareness'] + "□" * (5 - strategy['brand_awareness'])
                leads = "■" * strategy['lead_generation'] + "□" * (5 - strategy['lead_generation'])
                retain = "■" * strategy['customer_retention'] + "□" * (5 - strategy['customer_retention'])
                
                self.results_text.insert(tk.END, f"  Conversion Rate:      {conv} ({strategy['conversion_rate']}/5)\n", "info")
                self.results_text.insert(tk.END, f"  Brand Awareness:      {aware} ({strategy['brand_awareness']}/5)\n", "info")
                self.results_text.insert(tk.END, f"  Lead Generation:      {leads} ({strategy['lead_generation']}/5)\n", "info")
                self.results_text.insert(tk.END, f"  Customer Retention:   {retain} ({strategy['customer_retention']}/5)\n", "info")
                
                # Add separator between recommendations
                if i < len(recommendations):
                    self.results_text.insert(tk.END, "\n" + "─" * 50 + "\n\n")
            
            # Update status
            self.status_var.set(f"Successfully generated {len(recommendations)} recommendations")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.status_var.set("Error occurred")
        finally:
            self.results_text.configure(state="disabled")

if __name__ == "__main__":
    root = tk.Tk()
    app = AccessibleFuturisticGUI(root)
    root.mainloop() 