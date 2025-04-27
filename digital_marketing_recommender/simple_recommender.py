import csv
import math
import os
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

class SimpleMarketingRecommender:
    def __init__(self, data_path='data/marketing_strategies.csv', trend_data_path='data/marketing_trends.csv'):
        self.data_path = data_path
        self.trend_data_path = trend_data_path
        self.strategies = []
        self.trend_data = []
        self.trend_models = {}
        self.load_data()
        self.load_trend_data()
        self.train_trend_models()
    
    def load_data(self):
        """Load marketing strategies data from CSV"""
        if not os.path.exists(self.data_path):
            print(f"Error: Data file not found at {self.data_path}")
            return
            
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                self.strategies = list(reader)
                print(f"Loaded {len(self.strategies)} strategies from {self.data_path}")
                
            # Convert numeric strings to integers
            numeric_fields = [
                'strategy_id', 'budget_required', 'technical_expertise', 
                'time_investment', 'conversion_rate', 'brand_awareness', 
                'lead_generation', 'customer_retention', 'target_audience_size'
            ]
            
            for strategy in self.strategies:
                for field in numeric_fields:
                    if field in strategy:
                        try:
                            strategy[field] = int(strategy[field])
                        except (ValueError, TypeError):
                            print(f"Warning: Could not convert {field} to integer for strategy {strategy.get('strategy_name', 'unknown')}")
                            strategy[field] = 0
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def load_trend_data(self):
        """Load historical trend data for marketing strategies"""
        if not os.path.exists(self.trend_data_path):
            print(f"Error: Trend data file not found at {self.trend_data_path}")
            return
            
        try:
            with open(self.trend_data_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                self.trend_data = list(reader)
                print(f"Loaded {len(self.trend_data)} trend data points from {self.trend_data_path}")
                
            # Convert numeric strings to floats
            for trend in self.trend_data:
                try:
                    trend['date'] = datetime.strptime(trend['date'], '%Y-%m-%d')
                    trend['effectiveness'] = float(trend['effectiveness'])
                    trend['cost_efficiency'] = float(trend['cost_efficiency'])
                    trend['adoption_rate'] = float(trend['adoption_rate'])
                    trend['strategy_id'] = int(trend['strategy_id'])
                except (ValueError, TypeError) as e:
                    print(f"Warning: Could not convert data for trend point: {e}")
                    continue
        except Exception as e:
            print(f"Error loading trend data: {e}")
    
    def train_trend_models(self):
        """Train regression models for each strategy's trends"""
        print("\n🔄 Training trend models...")
        trained_count = 0
        skipped_count = 0
        
        for strategy in self.strategies:
            try:
                strategy_id = int(strategy['strategy_id'])
                strategy_trends = [t for t in self.trend_data if int(t['strategy_id']) == strategy_id]
                
                if len(strategy_trends) < 2:
                    skipped_count += 1
                    continue
                    
                # Prepare data for training
                X = np.array([(t['date'] - min(t['date'] for t in strategy_trends)).days 
                             for t in strategy_trends]).reshape(-1, 1)
                y_effectiveness = np.array([t['effectiveness'] for t in strategy_trends])
                y_cost = np.array([t['cost_efficiency'] for t in strategy_trends])
                y_adoption = np.array([t['adoption_rate'] for t in strategy_trends])
                
                # Train models
                self.trend_models[strategy_id] = {
                    'effectiveness': LinearRegression().fit(X, y_effectiveness),
                    'cost_efficiency': LinearRegression().fit(X, y_cost),
                    'adoption_rate': LinearRegression().fit(X, y_adoption)
                }
                trained_count += 1
                
            except Exception as e:
                print(f"⚠️ Error training model for strategy {strategy_id}: {e}")
                skipped_count += 1
                continue
            
        print(f"✅ Trained models for {trained_count} strategies")
        if skipped_count > 0:
            print(f"⚠️ Skipped {skipped_count} strategies due to insufficient data or errors")
    
    def predict_trend_metrics(self, strategy_id, days_ahead=30):
        """Predict future trend metrics for a strategy"""
        if strategy_id not in self.trend_models:
            return None
            
        models = self.trend_models[strategy_id]
        latest_date = max(t['date'] for t in self.trend_data if t['strategy_id'] == strategy_id)
        days_since_start = (latest_date - min(t['date'] for t in self.trend_data 
                                            if t['strategy_id'] == strategy_id)).days
        
        # Predict for multiple time points to identify growth rate
        future_days = np.array([[days_since_start + i] for i in range(1, days_ahead + 1)])
        
        # Get predictions for all future days
        effectiveness_predictions = models['effectiveness'].predict(future_days)
        cost_predictions = models['cost_efficiency'].predict(future_days)
        adoption_predictions = models['adoption_rate'].predict(future_days)
        
        # Calculate growth rates
        initial_effectiveness = effectiveness_predictions[0]
        final_effectiveness = effectiveness_predictions[-1]
        growth_rate = ((final_effectiveness - initial_effectiveness) / initial_effectiveness) * 100
        
        # Calculate acceleration (second derivative)
        effectiveness_acceleration = np.gradient(np.gradient(effectiveness_predictions))
        is_accelerating = np.mean(effectiveness_acceleration) > 0
        
        # Find when the strategy becomes trending (effectiveness crosses threshold)
        trending_threshold = 0.5  # Lower threshold to catch more potential trends
        trending_week = None
        for i, pred in enumerate(effectiveness_predictions):
            if pred >= trending_threshold:
                trending_week = (i + 1) // 7  # Convert days to weeks
                break
        
        # Determine if strategy is trending based on multiple factors
        is_trending = (
            effectiveness_predictions[-1] >= trending_threshold and
            growth_rate > 2.0 and  # Lower minimum growth rate
            adoption_predictions[-1] > 0.4  # Lower minimum adoption rate
        )
        
        return {
            'effectiveness': effectiveness_predictions[-1],
            'cost_efficiency': cost_predictions[-1],
            'adoption_rate': adoption_predictions[-1],
            'growth_rate': growth_rate,
            'trending_week': trending_week,
            'is_trending': is_trending,
            'acceleration': np.mean(effectiveness_acceleration)
        }
    
    def cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        # Calculate magnitudes
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
            
        # Return cosine similarity
        return dot_product / (magnitude1 * magnitude2)
    
    def get_recommendations(self, user_preferences, top_n=5, include_trends=True):
        """
        Get personalized recommendations based on user preferences and trend analysis
        
        Args:
            user_preferences: dict with the following keys:
                - budget_amount: monthly budget amount in dollars
                - daily_hours: number of hours available per day
                - technical_skill: 1-5 (1=beginner, 5=expert)
                - goal: one of ['conversion', 'awareness', 'leads', 'retention']
                - industry: string with the industry name
                - audience_size: 1-5 (1=very small, 5=very large)
            top_n: number of recommendations to return
            include_trends: whether to include trend predictions in recommendations
            
        Returns:
            List of recommended strategies with trend predictions
        """
        # Create user profile vector
        user_profile = [0] * 8
        
        # Convert budget amount to a level (1-5)
        budget_amount = user_preferences.get('budget_amount', 2000)
        if budget_amount < 500:
            budget_level = 1
        elif budget_amount < 1000:
            budget_level = 2
        elif budget_amount < 2000:
            budget_level = 3
        elif budget_amount < 5000:
            budget_level = 4
        else:
            budget_level = 5
            
        # Map budget to budget_required (inverse relationship)
        user_profile[0] = 6 - budget_level  # Inverse: higher budget = lower concern
        
        # Map technical skill to technical_expertise (inverse relationship)
        user_profile[1] = 6 - user_preferences['technical_skill']  # Inverse: higher skill = lower concern
        
        # Map daily hours to time_investment (inverse relationship)
        daily_hours = user_preferences.get('daily_hours', 8)
        if daily_hours < 2:
            time_level = 1
        elif daily_hours < 4:
            time_level = 2
        elif daily_hours < 6:
            time_level = 3
        elif daily_hours < 8:
            time_level = 4
        else:
            time_level = 5
        user_profile[2] = 6 - time_level  # Inverse: more time = lower concern
        
        # Map goal to specific features
        goal_mapping = {
            'conversion': 3,  # index of conversion_rate
            'awareness': 4,   # index of brand_awareness
            'leads': 5,       # index of lead_generation
            'retention': 6    # index of customer_retention
        }
        
        # Set default values for goals
        for i in range(3, 7):
            user_profile[i] = 3  # Set default value
            
        # Emphasize the specific goal
        if user_preferences['goal'] in goal_mapping:
            goal_index = goal_mapping[user_preferences['goal']]
            user_profile[goal_index] = 5  # Boost the specific goal
        
        # Map audience size
        user_profile[7] = user_preferences['audience_size']
        
        # Calculate similarity with all strategies
        strategy_similarities = []
        
        for strategy in self.strategies:
            # Extract strategy features
            strategy_features = [
                strategy['budget_required'],
                strategy['technical_expertise'],
                strategy['time_investment'],
                strategy['conversion_rate'],
                strategy['brand_awareness'],
                strategy['lead_generation'],
                strategy['customer_retention'],
                strategy['target_audience_size']
            ]
            
            # Calculate similarity
            similarity = self.cosine_similarity(user_profile, strategy_features)
            
            # Check if industry matches
            industry_match = False
            if user_preferences.get('industry'):
                industries = strategy['best_for_industry'].replace('"', '').split(',')
                industry_match = user_preferences['industry'] in industries or 'All' in industries
            else:
                industry_match = True  # No industry filter
                
            # Add to results if industry matches
            if industry_match:
                strategy_similarities.append((strategy, similarity))
        
        # Sort by similarity score (descending)
        strategy_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top N recommendations
        recommendations = strategy_similarities[:top_n]
        
        if include_trends:
            # Add trend predictions to each recommendation
            enhanced_recommendations = []
            for strategy, similarity in recommendations:
                trend_metrics = self.predict_trend_metrics(strategy['strategy_id'])
                if trend_metrics:
                    strategy['trend_metrics'] = trend_metrics
                    # Add trending status
                    strategy['is_trending'] = trend_metrics['is_trending']
                    strategy['trending_week'] = trend_metrics['trending_week']
                    strategy['growth_rate'] = trend_metrics['growth_rate']
                    strategy['acceleration'] = trend_metrics['acceleration']
                enhanced_recommendations.append((strategy, similarity))
            
            # Sort by growth rate and acceleration
            enhanced_recommendations.sort(
                key=lambda x: (
                    x[0].get('growth_rate', 0) * 0.7 + 
                    x[0].get('acceleration', 0) * 0.3
                ),
                reverse=True
            )
            
            return enhanced_recommendations
        
        return recommendations
    
    def get_strategy_details(self, strategy_id):
        """Get detailed information about a specific strategy"""
        for strategy in self.strategies:
            if strategy['strategy_id'] == strategy_id:
                return strategy
        return None

# Example usage
if __name__ == "__main__":
    recommender = SimpleMarketingRecommender()
    
    user_prefs = {
        'budget_amount': 1500,    # Medium budget
        'daily_hours': 4,          # Good amount of time
        'technical_skill': 2,      # Low technical skills
        'goal': 'awareness',       # Primary goal is brand awareness
        'industry': 'Fashion',     # Fashion industry
        'audience_size': 4         # Large audience
    }
    
    recommendations = recommender.get_recommendations(user_prefs, top_n=3)
    
    print("\nTop Recommendations:")
    for i, (strategy, similarity) in enumerate(recommendations, 1):
        print(f"{i}. {strategy['strategy_name']} (Similarity: {similarity:.2f})")
        print(f"   Best for: {strategy['best_for_industry']}")
        print(f"   Budget required: {strategy['budget_required']}/5")
        print(f"   Technical expertise: {strategy['technical_expertise']}/5")
        print(f"   Time investment: {strategy['time_investment']}/5")
        print(f"   Brand awareness impact: {strategy['brand_awareness']}/5")
        print()