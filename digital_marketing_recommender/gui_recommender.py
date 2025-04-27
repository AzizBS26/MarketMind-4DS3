import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from simple_recommender import SimpleMarketingRecommender

class MarketingRecommenderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🚀 Digital Marketing Strategy Recommender for SMEs")
        self.root.geometry("960x720")
        self.root.configure(bg="#E8F0FE")
        
        self.recommender = SimpleMarketingRecommender()
        if not self.recommender.strategies:
            messagebox.showerror("Error", "Failed to load marketing strategies data.")
            self.root.destroy()
            return

        self.all_industries = self.extract_all_industries()
        self.configure_styles()
        self.create_widgets()

    def extract_all_industries(self):
        all_industries = set()
        for strategy in self.recommender.strategies:
            industries = strategy['best_for_industry'].replace('"', '').split(',')
            all_industries.update(industries)
        return sorted(list(all_industries))

    def configure_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("TFrame", background="#E8F0FE")
        style.configure("TLabel", background="#E8F0FE", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10), padding=6)
        style.configure("Accent.TButton", font=("Segoe UI", 11, "bold"), foreground="white", background="#1976D2")
        style.map("Accent.TButton", background=[("active", "#1565C0")])

        style.configure("Header.TLabel", font=("Segoe UI", 16, "bold"), background="#E8F0FE")
        style.configure("SubHeader.TLabel", font=("Segoe UI", 10, "italic"), background="#E8F0FE")

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X)
        ttk.Label(header_frame, text="Digital Marketing Strategy Recommender for SMEs", style="Header.TLabel").pack()
        ttk.Label(header_frame, text="🧠 Find the most suitable strategies for your business needs", style="SubHeader.TLabel").pack(pady=5)

        form_frame = ttk.LabelFrame(main_frame, text="📝 Your Business Profile", padding=15)
        form_frame.pack(fill=tk.X, pady=15)

        self.industry_var = tk.StringVar()
        self.budget_amount_var = tk.IntVar(value=2000)
        self.daily_hours_var = tk.IntVar(value=8)
        self.tech_var = tk.IntVar(value=3)
        self.audience_var = tk.IntVar(value=3)
        self.goal_var = tk.StringVar(value="awareness")

        fields = [
            ("Industry:", ttk.Combobox(form_frame, textvariable=self.industry_var, width=30, values=self.all_industries)),
            ("Monthly Budget Amount ($):", ttk.Entry(form_frame, textvariable=self.budget_amount_var, width=15)),
            ("Daily Hours Available:", ttk.Entry(form_frame, textvariable=self.daily_hours_var, width=15)),
        ]

        for idx, (label, widget) in enumerate(fields):
            ttk.Label(form_frame, text=label).grid(row=idx, column=0, sticky=tk.W, pady=5)
            widget.grid(row=idx, column=1, sticky=tk.W, pady=5)

        # Technical skill slider
        ttk.Label(form_frame, text="Technical Skills (1=Beginner, 5=Expert):").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Scale(form_frame, from_=1, to=5, orient=tk.HORIZONTAL, variable=self.tech_var, length=200).grid(row=3, column=1, sticky=tk.W)

        # Audience size
        ttk.Label(form_frame, text="Target Audience Size (1=Small, 5=Large):").grid(row=4, column=0, sticky=tk.W, pady=5)
        ttk.Scale(form_frame, from_=1, to=5, orient=tk.HORIZONTAL, variable=self.audience_var, length=200).grid(row=4, column=1, sticky=tk.W)

        # Marketing goal radio buttons
        ttk.Label(form_frame, text="Primary Marketing Goal:").grid(row=5, column=0, sticky=tk.W, pady=5)
        goal_frame = ttk.Frame(form_frame)
        goal_frame.grid(row=5, column=1, sticky=tk.W, pady=5)

        goals = [
            ("Increase Conversion", "conversion"),
            ("Boost Awareness", "awareness"),
            ("Generate Leads", "leads"),
            ("Improve Retention", "retention")
        ]
        for i, (text, value) in enumerate(goals):
            ttk.Radiobutton(goal_frame, text=text, variable=self.goal_var, value=value).pack(side=tk.LEFT, padx=5)

        # Button
        ttk.Button(main_frame, text="🚀 Generate Recommendations", command=self.generate_recommendations, style="Accent.TButton").pack(pady=15)

        # Results box
        self.results_frame = ttk.LabelFrame(main_frame, text="📋 Recommended Strategies", padding=10)
        self.results_frame.pack(fill=tk.BOTH, expand=True)
        self.results_text = scrolledtext.ScrolledText(self.results_frame, wrap=tk.WORD, font=("Consolas", 10), height=20)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        self.results_text.insert(tk.END, "Please fill out your profile and click 'Generate Recommendations'.")
        self.results_text.config(state=tk.DISABLED)

        # Status
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X)

    def generate_recommendations(self):
        try:
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            self.status_var.set("⏳ Generating recommendations...")
            self.root.update_idletasks()

            industry = self.industry_var.get()
            if not industry:
                messagebox.showwarning("Input Error", "Please select an industry.")
                self.status_var.set("⚠️ Input missing")
                return

            prefs = {
                'budget_amount': self.budget_amount_var.get(),
                'daily_hours': self.daily_hours_var.get(),
                'technical_skill': self.tech_var.get(),
                'goal': self.goal_var.get(),
                'industry': industry,
                'audience_size': self.audience_var.get()
            }

            recommendations = self.recommender.get_recommendations(prefs, top_n=3, include_trends=True)
            if not recommendations:
                self.results_text.insert(tk.END, "❌ No matching strategies found.")
                self.status_var.set("No matches found")
                return

            self.results_text.insert(tk.END, "🎯 TOP MARKETING STRATEGIES\n" + "=" * 60 + "\n\n")
            for i, (strategy, similarity) in enumerate(recommendations, 1):
                self.results_text.insert(tk.END, f"#{i} {strategy['strategy_name']} (Match: {similarity:.2f})\n")
                self.results_text.insert(tk.END, f"- Best for: {strategy['best_for_industry']}\n")
                self.results_text.insert(tk.END, f"- Budget: {'$' * strategy['budget_required']} ({strategy['budget_required']}/5)\n")
                self.results_text.insert(tk.END, f"- Expertise: {'*' * strategy['technical_expertise']} ({strategy['technical_expertise']}/5)\n")
                self.results_text.insert(tk.END, f"- Time: {'⏱' * strategy['time_investment']} ({strategy['time_investment']}/5)\n")
                self.results_text.insert(tk.END, f"- Conversion: {'↑' * strategy['conversion_rate']} ({strategy['conversion_rate']}/5)\n")
                self.results_text.insert(tk.END, f"- Awareness: {'👁' * strategy['brand_awareness']} ({strategy['brand_awareness']}/5)\n")
                self.results_text.insert(tk.END, f"- Leads: {'⚡' * strategy['lead_generation']} ({strategy['lead_generation']}/5)\n")
                self.results_text.insert(tk.END, f"- Retention: {'♥' * strategy['customer_retention']} ({strategy['customer_retention']}/5)\n")
                if 'trend_metrics' in strategy:
                    self.results_text.insert(tk.END, f"- Trend Effectiveness: {strategy['trend_metrics']['effectiveness']:.2f}\n")
                    self.results_text.insert(tk.END, f"- Cost Efficiency: {strategy['trend_metrics']['cost_efficiency']:.2f}\n")
                    self.results_text.insert(tk.END, f"- Adoption Rate: {strategy['trend_metrics']['adoption_rate']:.2f}\n")
                self.results_text.insert(tk.END, "-" * 60 + "\n\n")

            self.status_var.set(f"✅ {len(recommendations)} recommendations generated")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_var.set("❌ Error occurred")
        finally:
            self.results_text.config(state=tk.DISABLED)


if __name__ == "__main__":
    root = tk.Tk()
    app = MarketingRecommenderGUI(root)
    root.mainloop()
