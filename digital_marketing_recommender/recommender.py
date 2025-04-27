import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

class MarketingRecommender:
    def __init__(self, data_path='data/marketing_strategies.csv'):
        self.data_path = data_path
        self.strategies_df = None
        self.features_matrix = None
        self.scaler = MinMaxScaler()
        self.similarity_matrix = None
        self.load_data()
        self.preprocess_data()
        
    def load_data(self):
        """Load marketing strategies data"""
        self.strategies_df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.strategies_df)} marketing strategies")
        
    def preprocess_data(self):
        """Preprocess data for the recommendation engine"""
        # Extract numeric features for similarity calculation
        numeric_features = [
            'budget_required', 'technical_expertise', 'time_investment',
            'conversion_rate', 'brand_awareness', 'lead_generation', 
            'customer_retention', 'target_audience_size'
        ]
        
        # Normalize features
        self.features_matrix = self.scaler.fit_transform(
            self.strategies_df[numeric_features]
        )
        
        # Calculate similarity matrix
        self.similarity_matrix = cosine_similarity(self.features_matrix)
        
    def get_recommendations(self, user_preferences, top_n=3):
        """
        Get personalized recommendations based on user preferences
        
        Args:
            user_preferences: dict with the following keys:
                - budget_amount: default budget amount in dollars (2000)
                - daily_hours: number of hours available per day
                - technical_skill: 1-5 (1=beginner, 5=expert)
                - goal: one of ['conversion', 'awareness', 'leads', 'retention']
                - industry: string with the industry name
                - audience_size: 1-5 (1=very small, 5=very large)
            top_n: number of recommendations to return
            
        Returns:
            DataFrame with recommended strategies
        """
        # Create a user profile vector
        user_profile = np.zeros(self.features_matrix.shape[1])
        
        # Map technical skill to technical_expertise (inverse relationship)
        user_profile[1] = 6 - user_preferences['technical_skill']  # Inverse: higher skill = lower concern
        
        # Map goal to specific features
        goal_mapping = {
            'conversion': 3,  # index of conversion_rate
            'awareness': 4,   # index of brand_awareness
            'leads': 5,       # index of lead_generation
            'retention': 6    # index of customer_retention
        }
        
        # Emphasize the specific goal
        for i in range(3, 7):
            user_profile[i] = 3  # Set default value
        if user_preferences['goal'] in goal_mapping:
            goal_index = goal_mapping[user_preferences['goal']]
            user_profile[goal_index] = 5  # Boost the specific goal
        
        # Map audience size
        user_profile[7] = user_preferences['audience_size']
        
        # Normalize the user profile
        user_profile_scaled = self.scaler.transform([user_profile])[0]
        
        # Calculate similarity with all strategies
        similarities = cosine_similarity([user_profile_scaled], self.features_matrix)[0]
        
        # Create a copy of the strategies dataframe
        result_df = self.strategies_df.copy()
        
        # Add similarity score
        result_df['similarity_score'] = similarities
        
        # Filter by industry if specified
        if user_preferences.get('industry'):
            # Filter strategies that work for the specified industry
            industry_mask = result_df['best_for_industry'].apply(
                lambda x: user_preferences['industry'] in x or 'All' in x
            )
            result_df = result_df[industry_mask]
            
        # Filter by budget amount
        budget_mask = result_df['estimated_cost'] <= user_preferences['budget_amount']
        result_df = result_df[budget_mask]
            
        # Filter by daily hours if specified
        if 'daily_hours' in user_preferences:
            hours_mask = result_df['daily_time_requirement'] <= user_preferences['daily_hours']
            result_df = result_df[hours_mask]
        
        # Sort by similarity score and return top_n recommendations
        recommendations = result_df.sort_values(
            by='similarity_score', ascending=False
        ).head(top_n)
        
        return recommendations[['strategy_id', 'strategy_name', 'similarity_score']]
    
    def get_strategy_details(self, strategy_id):
        """Get detailed information about a specific strategy"""
        return self.strategies_df[self.strategies_df['strategy_id'] == strategy_id].iloc[0].to_dict()
    
    def save_model(self, path='model.joblib'):
        """Save the recommender model"""
        model_data = {
            'strategies_df': self.strategies_df,
            'features_matrix': self.features_matrix,
            'scaler': self.scaler,
            'similarity_matrix': self.similarity_matrix
        }
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load_model(cls, path='model.joblib'):
        """Load a saved recommender model"""
        model_data = joblib.load(path)
        recommender = cls.__new__(cls)
        recommender.strategies_df = model_data['strategies_df']
        recommender.features_matrix = model_data['features_matrix']
        recommender.scaler = model_data['scaler']
        recommender.similarity_matrix = model_data['similarity_matrix']
        recommender.data_path = None
        return recommender


if __name__ == "__main__":
    # Example usage
    recommender = MarketingRecommender()
    
    user_prefs = {
        'daily_hours': 8,          # 8 hours available per day
        'technical_skill': 2,       # Low technical skills
        'goal': 'awareness',        # Primary goal is brand awareness
        'industry': 'Fashion',      # Fashion industry
        'audience_size': 4          # Large audience
    }
    
    recommendations = recommender.get_recommendations(user_prefs, top_n=3)
    print("\nTop Recommendations:")
    print(recommendations)
    
    # Save the model
    recommender.save_model('data/marketing_recommender_model.joblib')