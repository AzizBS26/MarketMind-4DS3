#!/usr/bin/env python3
"""
Product Trend Prediction System

This system predicts which products will trend in the future based on client inputs
and historical trend data.
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import random

# Ensure correct path resolution
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

class ProductTrendPredictor:
    """Predicts product trends based on customer inputs and historical data"""
    
    def __init__(self, product_data_path=None):
        """Initialize the product trend predictor with product data"""
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Use provided path or default to the built-in product dataset
        if product_data_path is None:
            product_data_path = os.path.join(self.script_dir, "data", "product_trends.csv")
            
            # If product data doesn't exist, generate synthetic data
            if not os.path.exists(product_data_path):
                print(f"Product data not found at {product_data_path}")
                print("Generating synthetic product trend data...")
                self.generate_product_data(product_data_path)
        
        # Load product data
        if os.path.exists(product_data_path):
            self.product_data = pd.read_csv(product_data_path)
            print(f"✅ Loaded {len(self.product_data)} products from {product_data_path}")
        else:
            raise FileNotFoundError(f"Product data file not found: {product_data_path}")
        
        # Initialize trend models
        self.trend_models = {}
        self.train_trend_models()
    
    def generate_product_data(self, output_path):
        """Generate synthetic product data with trend information"""
        # Product categories
        categories = [
            "Clothing", "Electronics", "Home Decor", "Beauty", "Health", 
            "Fitness", "Kitchen", "Office", "Outdoors", "Pet Supplies",
            "Food & Beverage", "Toys", "Books", "Art Supplies", "Handmade"
        ]
        
        # Price ranges by category
        price_ranges = {
            "Clothing": (15, 200),
            "Electronics": (50, 2000),
            "Home Decor": (20, 500),
            "Beauty": (10, 150),
            "Health": (5, 100),
            "Fitness": (15, 300),
            "Kitchen": (10, 400),
            "Office": (5, 200),
            "Outdoors": (20, 500),
            "Pet Supplies": (5, 100),
            "Food & Beverage": (5, 50),
            "Toys": (10, 100),
            "Books": (10, 50),
            "Art Supplies": (5, 100),
            "Handmade": (20, 300)
        }
        
        # Target demographics
        demographics = [
            "Young Adults", "Professionals", "Parents", "Seniors", "Teenagers",
            "Children", "Men", "Women", "Families", "Students"
        ]
        
        # Product name components
        adjectives = [
            "Premium", "Eco-Friendly", "Smart", "Handcrafted", "Vintage", 
            "Modern", "Organic", "Luxury", "Budget", "Professional",
            "Compact", "Portable", "Durable", "Lightweight", "Adjustable"
        ]
        
        objects = {
            "Clothing": ["T-Shirt", "Jeans", "Dress", "Jacket", "Sweater", "Hat", "Socks", "Scarf"],
            "Electronics": ["Headphones", "Speaker", "Charger", "Tablet", "Phone Case", "Smartwatch", "Camera"],
            "Home Decor": ["Pillow", "Lamp", "Vase", "Frame", "Rug", "Blanket", "Mirror", "Clock"],
            "Beauty": ["Moisturizer", "Serum", "Mask", "Lipstick", "Eyeshadow", "Cleanser", "Brush Set"],
            "Health": ["Vitamins", "Supplements", "Essential Oils", "Tea", "First Aid Kit", "Massager"],
            "Fitness": ["Yoga Mat", "Dumbbells", "Resistance Bands", "Water Bottle", "Tracker", "Bag"],
            "Kitchen": ["Knife Set", "Blender", "Cookware", "Utensils", "Storage Containers", "Cutting Board"],
            "Office": ["Notebook", "Pen Set", "Desk Organizer", "Planner", "Laptop Stand", "Chair Cushion"],
            "Outdoors": ["Backpack", "Tent", "Camping Gear", "Hiking Boots", "Binoculars", "Hammock"],
            "Pet Supplies": ["Pet Bed", "Toy", "Bowl", "Collar", "Leash", "Carrier", "Treats"],
            "Food & Beverage": ["Coffee", "Tea", "Spices", "Chocolate", "Snacks", "Gourmet Set"],
            "Toys": ["Building Blocks", "Puzzle", "Action Figure", "Plush Toy", "Board Game", "Craft Kit"],
            "Books": ["Fiction", "Non-Fiction", "Cookbook", "Journal", "Coloring Book", "Reference Guide"],
            "Art Supplies": ["Paint Set", "Sketchbook", "Canvas", "Brush Set", "Markers", "Clay Kit"],
            "Handmade": ["Jewelry", "Candle", "Soap", "Wall Art", "Planter", "Coasters", "Keychain"]
        }
        
        brands = {
            "Clothing": ["StyleCraft", "UrbanThread", "EcoWear", "ModernChic", "ComfortZone"],
            "Electronics": ["TechPro", "SoundWave", "SmartLife", "DigitalEdge", "PowerTech"],
            "Home Decor": ["HomeEssence", "CozyNest", "ElegantLiving", "ModernSpace", "NaturalHome"],
            "Beauty": ["GlowUp", "NaturalGlow", "BeautyEssence", "PureRadiance", "LuxeBeauty"],
            "Health": ["VitalWell", "NatureBoost", "PureHealth", "HolisticLife", "WellnessDaily"],
            "Fitness": ["ActiveLife", "FitZone", "PowerMove", "FlexFit", "EnduranceMax"],
            "Kitchen": ["ChefChoice", "KitchenPro", "CookSmart", "GourmetBasics", "CulinaryEdge"],
            "Office": ["WorkSmart", "OfficePro", "ProductivityPlus", "DeskMaster", "OrganizeIt"],
            "Outdoors": ["WildTrek", "NatureExplorer", "AdventureGear", "OutdoorLife", "TrailBlazer"],
            "Pet Supplies": ["PetLove", "HappyPets", "WagWell", "FurFriends", "PetComfort"],
            "Food & Beverage": ["TastyBites", "FoodCraft", "FlavorsDelight", "GourmetSelect", "CulinaryJoy"],
            "Toys": ["FunZone", "PlayTime", "ImagineCraft", "KidJoy", "CreativePlay"],
            "Books": ["PageTurner", "MindJourney", "ReadMore", "BookWorm", "StoryWorld"],
            "Art Supplies": ["CreativeSpirit", "ArtistChoice", "ColorCraft", "SketchMaster", "ArtEssentials"],
            "Handmade": ["CraftedJoy", "HandmadeWonder", "ArtisanCreations", "UniqueByHand", "CraftedLife"]
        }
        
        # Generate random products
        num_products = 200
        products = []
        
        # Start with a recent date and generate historic data from there
        end_date = datetime.now() - timedelta(days=7)
        
        for i in range(1, num_products + 1):
            # Select random category
            category = random.choice(categories)
            
            # Generate product name
            adjective = random.choice(adjectives)
            obj = random.choice(objects[category])
            brand = random.choice(brands.get(category, ["Generic"]))
            product_name = f"{adjective} {obj} by {brand}"
            
            # Set price
            min_price, max_price = price_ranges.get(category, (10, 100))
            price = round(random.uniform(min_price, max_price), 2)
            
            # Target demographic
            target_demo = random.sample(demographics, k=random.randint(1, 3))
            target_demographic = ", ".join(target_demo)
            
            # Seasonal nature (higher is more seasonal)
            seasonality = random.randint(1, 5)
            
            # Product age (months)
            product_age = random.randint(1, 36)
            
            # Initial popularity (1-100)
            initial_popularity = random.randint(20, 90)
            
            # Growth potential (1-100)
            if random.random() < 0.15:  # 15% high growth
                growth_potential = random.randint(70, 100)
            elif random.random() < 0.55:  # 40% medium growth
                growth_potential = random.randint(40, 69)
            else:  # 45% low growth
                growth_potential = random.randint(10, 39)
            
            # Current trend value
            trend_value = initial_popularity + random.randint(-10, 20)
            if trend_value < 0:
                trend_value = 0
            elif trend_value > 100:
                trend_value = 100
            
            # Social media mentions (weekly)
            social_mentions = int(trend_value * random.uniform(1, 5))
            
            # Search volume (weekly)
            search_volume = int(trend_value * random.uniform(2, 10))
            
            # Generate trend history (weekly data points for the last 12 weeks)
            trend_history = []
            current_trend = max(0, min(100, initial_popularity - 20))
            
            for week in range(12, 0, -1):
                date = end_date - timedelta(days=7 * week)
                
                # Add some randomness and seasonal effects
                season_effect = 0
                # Simple seasonal effect based on month
                month = date.month
                if seasonality > 3:  # Only high seasonality products affected
                    # Summer peak (June-August)
                    if 6 <= month <= 8:
                        season_effect = random.randint(5, 15) if category in ["Outdoors", "Fitness", "Clothing"] else 0
                    # Winter peak (November-January)
                    elif month in [11, 12, 1]:
                        season_effect = random.randint(5, 15) if category in ["Clothing", "Home Decor", "Food & Beverage"] else 0
                
                # Random weekly fluctuation
                random_effect = random.randint(-5, 8)
                
                # Growth trend
                growth_effect = (growth_potential / 100) * (12 - week) / 3
                
                # Combined effect
                current_trend += growth_effect + season_effect + random_effect
                current_trend = max(0, min(100, current_trend))
                
                trend_history.append({
                    "product_id": i,
                    "date": date.strftime("%Y-%m-%d"),
                    "trend_score": round(current_trend, 1),
                    "social_mentions": int(current_trend * random.uniform(1, 5)),
                    "search_volume": int(current_trend * random.uniform(2, 10))
                })
            
            # Final product record
            product = {
                "product_id": i,
                "product_name": product_name,
                "category": category,
                "price": price,
                "target_demographic": target_demographic,
                "seasonality": seasonality,
                "product_age_months": product_age,
                "growth_potential": growth_potential,
                "current_trend_value": trend_value,
                "social_mentions_weekly": social_mentions,
                "search_volume_weekly": search_volume
            }
            
            products.append(product)
        
        # Convert to DataFrame
        products_df = pd.DataFrame(products)
        
        # Create trend history DataFrame
        trend_history_df = pd.DataFrame([record for product in trend_history for record in product] 
                                        if trend_history else [])
        
        # Save product data
        products_df.to_csv(output_path, index=False)
        
        # Save trend history to a separate file
        trend_history_path = output_path.replace(".csv", "_history.csv")
        if trend_history_df.empty:
            trend_history_df = pd.DataFrame(columns=["product_id", "date", "trend_score", 
                                                    "social_mentions", "search_volume"])
        else:
            trend_history_df.to_csv(trend_history_path, index=False)
        
        # Return the generated data
        self.product_data = products_df
        return products_df
    
    def train_trend_models(self):
        """Train trend prediction models for each product"""
        # In a real implementation, this would load historical trend data and train models
        # For this example, we'll use a simplified approach
        
        print("Training product trend prediction models...")
        
        # For each product, create a simple trend model
        for _, product in self.product_data.iterrows():
            product_id = product['product_id']
            growth_potential = product.get('growth_potential', 50)
            
            # Simple model: Just use growth potential with some randomness
            # In a real implementation, this would be a trained machine learning model
            self.trend_models[product_id] = {
                'growth_rate': growth_potential / 100 * random.uniform(0.8, 1.2),
                'base_value': product.get('current_trend_value', 50),
                'seasonality': product.get('seasonality', 1)
            }
        
        print(f"✅ Trained trend models for {len(self.trend_models)} products")
    
    def predict_trending_products(self, client_inputs, max_results=5):
        """
        Predict which products will trend based on client inputs
        
        Args:
            client_inputs: Dict with client inputs like:
                - budget_range: (min, max) price range
                - target_demographic: Target audience
                - categories: List of product categories of interest
                - trend_window_weeks: How far ahead to predict (default: 4 weeks)
            max_results: Maximum number of products to return
            
        Returns:
            List of product dictionaries with trend predictions
        """
        # Extract client inputs
        budget_min = client_inputs.get('budget_min', 0)
        budget_max = client_inputs.get('budget_max', 10000)
        target_demographic = client_inputs.get('target_demographic', None)
        categories = client_inputs.get('categories', None)
        trend_window_weeks = client_inputs.get('trend_window_weeks', 4)
        
        # Filter products by client inputs
        filtered_products = self.product_data.copy()
        
        # Filter by price range
        filtered_products = filtered_products[
            (filtered_products['price'] >= budget_min) & 
            (filtered_products['price'] <= budget_max)
        ]
        
        # Filter by category if specified
        if categories and len(categories) > 0:
            filtered_products = filtered_products[filtered_products['category'].isin(categories)]
        
        # Filter by demographic if specified
        if target_demographic:
            # This is a simplified approach - in reality would need more sophisticated matching
            filtered_products = filtered_products[
                filtered_products['target_demographic'].str.contains(target_demographic, case=False, na=False)
            ]
        
        # If no products match the criteria, return empty list
        if len(filtered_products) == 0:
            return []
        
        # Calculate trend predictions for each product
        predictions = []
        
        for _, product in filtered_products.iterrows():
            product_id = product['product_id']
            
            # Skip if no model for this product
            if product_id not in self.trend_models:
                continue
            
            # Get model parameters
            model = self.trend_models[product_id]
            growth_rate = model['growth_rate']
            base_value = model['base_value']
            seasonality = model['seasonality']
            
            # Current date for seasonal effects
            current_date = datetime.now()
            future_date = current_date + timedelta(weeks=trend_window_weeks)
            
            # Simple seasonal effect
            season_effect = 0
            if seasonality >= 3:  # Only high seasonality products affected
                # Summer peak (June-August)
                if 6 <= future_date.month <= 8:
                    season_effect = 10 if product['category'] in ["Outdoors", "Fitness", "Clothing"] else 0
                # Winter peak (November-January)
                elif future_date.month in [11, 12, 1]:
                    season_effect = 10 if product['category'] in ["Clothing", "Home Decor", "Food & Beverage"] else 0
            
            # Calculate predicted trend value
            current_trend = base_value
            predicted_trend = current_trend + (growth_rate * trend_window_weeks * 10) + season_effect
            predicted_trend = max(0, min(100, predicted_trend))
            
            # Calculate growth percentage
            growth_percentage = ((predicted_trend - current_trend) / current_trend * 100) if current_trend > 0 else 0
            
            # Create prediction record
            prediction = product.to_dict()
            prediction['predicted_trend'] = round(predicted_trend, 1)
            prediction['growth_percentage'] = round(growth_percentage, 1)
            prediction['trend_window_weeks'] = trend_window_weeks
            
            predictions.append(prediction)
        
        # Sort by predicted trend value (descending)
        predictions.sort(key=lambda x: x['predicted_trend'], reverse=True)
        
        # Return top N results
        return predictions[:max_results]
    
    def get_product_recommendations(self, client_inputs):
        """
        Get product recommendations with additional context
        
        Args:
            client_inputs: Dict with client preferences
            
        Returns:
            Dict with recommendations and summary
        """
        # Get trending product predictions
        trending_products = self.predict_trending_products(client_inputs)
        
        # If no products match, return empty response
        if not trending_products:
            return {
                "status": "no_results",
                "message": "No products match your criteria. Try broadening your search parameters.",
                "products": []
            }
        
        # Calculate average growth and trend values
        avg_growth = sum(p['growth_percentage'] for p in trending_products) / len(trending_products)
        avg_trend = sum(p['predicted_trend'] for p in trending_products) / len(trending_products)
        
        # Create recommendation context
        context = {}
        
        # Add time context
        trend_window = trending_products[0]['trend_window_weeks']
        future_date = datetime.now() + timedelta(weeks=trend_window)
        context['prediction_date'] = future_date.strftime("%B %Y")
        context['weeks_ahead'] = trend_window
        
        # Add category insights
        categories = {}
        for product in trending_products:
            cat = product['category']
            if cat not in categories:
                categories[cat] = 0
            categories[cat] += 1
        top_category = max(categories.items(), key=lambda x: x[1])[0] if categories else "N/A"
        context['top_category'] = top_category
        context['category_distribution'] = categories
        
        # Price insights
        prices = [p['price'] for p in trending_products]
        context['avg_price'] = round(sum(prices) / len(prices), 2)
        context['price_range'] = (min(prices), max(prices))
        
        # Growth insights
        context['avg_growth'] = round(avg_growth, 1)
        context['avg_trend_value'] = round(avg_trend, 1)
        
        # Return full recommendation package
        return {
            "status": "success",
            "message": f"Found {len(trending_products)} trending products for {context['prediction_date']}",
            "context": context,
            "products": trending_products
        }

# When run directly, test the functionality
if __name__ == "__main__":
    predictor = ProductTrendPredictor()
    
    # Test with sample client inputs
    client_inputs = {
        'budget_min': 20,
        'budget_max': 200,
        'target_demographic': 'Young Adults',
        'categories': ['Clothing', 'Electronics', 'Fitness'],
        'trend_window_weeks': 8
    }
    
    recommendations = predictor.get_product_recommendations(client_inputs)
    
    print("\n📊 Product Trend Predictions")
    print("=" * 60)
    
    if recommendations['status'] == 'success':
        context = recommendations['context']
        print(f"Prediction for: {context['prediction_date']} ({context['weeks_ahead']} weeks ahead)")
        print(f"Top category: {context['top_category']}")
        print(f"Average growth: {context['avg_growth']}%")
        print(f"Price range: ${context['price_range'][0]:.2f} - ${context['price_range'][1]:.2f}")
        print("\nTop Trending Products:")
        print("-" * 60)
        
        for i, product in enumerate(recommendations['products'], 1):
            print(f"{i}. {product['product_name']}")
            print(f"   Category: {product['category']}")
            print(f"   Price: ${product['price']:.2f}")
            print(f"   Target: {product['target_demographic']}")
            print(f"   Trend Score: {product['predicted_trend']}/100 (↑ {product['growth_percentage']}%)")
            print("-" * 60)
    else:
        print(f"No results: {recommendations['message']}") 