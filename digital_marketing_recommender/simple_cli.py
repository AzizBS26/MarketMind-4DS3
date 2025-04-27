import os
import sys
from simple_recommender import SimpleMarketingRecommender

def clear_screen():
    """Clear the console screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print the app header"""
    print("\n" + "=" * 80)
    print(" " * 20 + "DIGITAL MARKETING STRATEGY RECOMMENDER FOR SMEs")
    print("=" * 80 + "\n")
    print("This tool helps SMEs find the most suitable digital marketing strategies")
    print("based on their specific needs, resources, and goals.\n")

def get_integer_input(prompt, min_val, max_val):
    """Get an integer input from the user within a range"""
    while True:
        try:
            value = int(input(prompt))
            if min_val <= value <= max_val:
                return value
            print(f"Please enter a value between {min_val} and {max_val}.")
        except ValueError:
            print("Please enter a valid number.")

def get_industry_input(all_industries):
    """Get industry selection from the user"""
    print("\nAvailable Industries:")
    for i, industry in enumerate(all_industries, 1):
        print(f"{i}. {industry}")
    
    choice = get_integer_input("\nSelect your industry (number): ", 1, len(all_industries))
    return all_industries[choice - 1]

def get_goal_input():
    """Get marketing goal from the user"""
    goals = {
        1: ("conversion", "Increase Conversion Rates"),
        2: ("awareness", "Boost Brand Awareness"),
        3: ("leads", "Generate More Leads"),
        4: ("retention", "Improve Customer Retention")
    }
    
    print("\nMarketing Goals:")
    for num, (_, description) in goals.items():
        print(f"{num}. {description}")
    
    choice = get_integer_input("\nSelect your primary goal (number): ", 1, len(goals))
    return goals[choice][0]

def extract_all_industries(recommender):
    """Extract all unique industries from the dataset"""
    all_industries = set()
    for strategy in recommender.strategies:
        industries = strategy['best_for_industry'].replace('"', '').split(',')
        all_industries.update(industries)
    return sorted(list(all_industries))

def display_recommendations(recommendations):
    """Display recommended strategies"""
    print("\n" + "=" * 80)
    print(" " * 30 + "TOP RECOMMENDATIONS")
    print("=" * 80 + "\n")
    
    for i, (strategy, similarity) in enumerate(recommendations, 1):
        print(f"RECOMMENDATION #{i} - Match Score: {similarity:.2f}")
        print("-" * 60)
        print(f"Strategy: {strategy['strategy_name']}")
        print(f"Best for: {strategy['best_for_industry']}")
        print("\nResource Requirements:")
        print(f"  Budget required: {'$' * strategy['budget_required']} ({strategy['budget_required']}/5)")
        print(f"  Technical expertise: {'*' * strategy['technical_expertise']} ({strategy['technical_expertise']}/5)")
        print(f"  Time investment: {'⏱' * strategy['time_investment']} ({strategy['time_investment']}/5)")
        
        print("\nExpected Outcomes:")
        print(f"  Conversion Rate: {'↑' * strategy['conversion_rate']} ({strategy['conversion_rate']}/5)")
        print(f"  Brand Awareness: {'👁' * strategy['brand_awareness']} ({strategy['brand_awareness']}/5)")
        print(f"  Lead Generation: {'⚡' * strategy['lead_generation']} ({strategy['lead_generation']}/5)")
        print(f"  Customer Retention: {'♥' * strategy['customer_retention']} ({strategy['customer_retention']}/5)")
        print("\n" + "-" * 60 + "\n")
    
    input("Press Enter to continue...")

def main():
    """Main application function"""
    # Initialize the recommender
    recommender = SimpleMarketingRecommender()
    if not recommender.strategies:
        print("Failed to load marketing strategies data. Exiting...")
        sys.exit(1)
    
    # Extract all unique industries from the dataset
    all_industries = extract_all_industries(recommender)
    
    while True:
        clear_screen()
        print_header()
        
        print("Please provide information about your business to get personalized recommendations.\n")
        
        # Collect user preferences
        industry = get_industry_input(all_industries)
        
        print("\n" + "-" * 60)
        budget_level = get_integer_input(
            "Budget Level (1=Very Low, 5=Very High): ", 1, 5
        )
        
        technical_skill = get_integer_input(
            "Technical Skills (1=Beginner, 5=Expert): ", 1, 5
        )
        
        time_available = get_integer_input(
            "Time Available (1=Very Limited, 5=Abundant): ", 1, 5
        )
        
        audience_size = get_integer_input(
            "Target Audience Size (1=Very Small, 5=Very Large): ", 1, 5
        )
        
        goal = get_goal_input()
        
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
        clear_screen()
        print("Analyzing the best marketing strategies for your business...\n")
        recommendations = recommender.get_recommendations(user_prefs, top_n=3)
        
        # Display recommendations
        display_recommendations(recommendations)
        
        # Ask if the user wants to try again
        clear_screen()
        again = input("Would you like to get more recommendations? (y/n): ").lower()
        if again != 'y':
            break
    
    print("\nThank you for using the Digital Marketing Strategy Recommender!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram terminated by user.")
        sys.exit(0)