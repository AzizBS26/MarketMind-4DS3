import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from recommender import MarketingRecommender

# Set page config
st.set_page_config(
    page_title="Digital Marketing Strategy Recommender",
    page_icon="📊",
    layout="wide"
)

# Initialize recommender
@st.cache_resource
def load_recommender():
    model_path = os.path.join('data', 'marketing_recommender_model.joblib')
    if os.path.exists(model_path):
        # Load pre-trained model if available
        try:
            return MarketingRecommender.load_model(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            
    # Create a new model if no saved model exists
    recommender = MarketingRecommender()
    recommender.save_model(model_path)
    return recommender

# Extract all unique industries from the dataset
def get_all_industries(recommender):
    all_industries = []
    for industry_list in recommender.strategies_df['best_for_industry']:
        industries = industry_list.replace('"', '').split(',')
        all_industries.extend(industries)
    return sorted(list(set(all_industries)))

# Create radar chart for strategy visualization
def create_radar_chart(strategy_data):
    # Extract features for radar chart
    features = ['Budget', 'Tech Expertise', 'Time Investment', 
                'Conversion Rate', 'Brand Awareness', 
                'Lead Generation', 'Customer Retention']
    
    values = [
        strategy_data['budget_required'],
        strategy_data['technical_expertise'],
        strategy_data['time_investment'],
        strategy_data['conversion_rate'],
        strategy_data['brand_awareness'],
        strategy_data['lead_generation'],
        strategy_data['customer_retention']
    ]
    
    # Set up the radar chart
    angles = [n / float(len(features)) * 2 * 3.14159 for n in range(len(features))]
    angles += angles[:1]  # Close the loop
    values += values[:1]  # Close the loop
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, alpha=0.25)
    
    # Set labels and styling
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['1', '2', '3', '4', '5'])
    ax.set_ylim(0, 5)
    
    plt.title(strategy_data['strategy_name'], size=15, y=1.1)
    
    return fig

# Main App 
def main():
    # Sidebar for user input
    st.sidebar.title("SME Profile")
    
    # Get recommender
    recommender = load_recommender()
    
    # Get all industries
    all_industries = get_all_industries(recommender)
    
    # User inputs for recommendations
    st.sidebar.subheader("Business Details")
    
    industry = st.sidebar.selectbox(
        "Industry", 
        options=all_industries
    )
    
    audience_size = st.sidebar.slider(
        "Target Audience Size", 
        min_value=1, 
        max_value=5, 
        value=3,
        help="1=Very Small, 5=Very Large"
    )
    
    st.sidebar.subheader("Resources Available")
    
    budget_level = st.sidebar.slider(
        "Budget Level", 
        min_value=1, 
        max_value=5, 
        value=3,
        help="1=Very Low, 5=Very High"
    )
    
    technical_skill = st.sidebar.slider(
        "Technical Skills", 
        min_value=1, 
        max_value=5, 
        value=3,
        help="1=Beginner, 5=Expert"
    )
    
    time_available = st.sidebar.slider(
        "Time Available", 
        min_value=1, 
        max_value=5, 
        value=3,
        help="1=Very Limited, 5=Abundant"
    )
    
    st.sidebar.subheader("Marketing Goal")
    goal = st.sidebar.radio(
        "Primary Goal",
        options=["conversion", "awareness", "leads", "retention"],
        format_func=lambda x: {
            'conversion': 'Increase Conversion Rates',
            'awareness': 'Boost Brand Awareness', 
            'leads': 'Generate More Leads',
            'retention': 'Improve Customer Retention'
        }[x]
    )
    
    # Main content
    st.title("Digital Marketing Strategy Recommender for SMEs")
    st.write("""
    This recommendation system helps small and medium enterprises (SMEs) find the most suitable 
    digital marketing strategies based on their specific needs, resources, and goals.
    """)
    
    # User profile summary
    st.header("Your Business Profile")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Business Characteristics")
        st.write(f"**Industry:** {industry}")
        st.write(f"**Target Audience Size:** {audience_size}/5")
        st.write(f"**Primary Goal:** {goal.capitalize()}")
    
    with col2:
        st.subheader("Available Resources")
        st.write(f"**Budget Level:** {budget_level}/5")
        st.write(f"**Technical Skills:** {technical_skill}/5")
        st.write(f"**Time Available:** {time_available}/5")
    
    # Get recommendations button
    if st.button("Generate Recommendations"):
        # Create user preferences dictionary
        user_prefs = {
            'budget_level': budget_level,
            'technical_skill': technical_skill,
            'time_available': time_available,
            'goal': goal,
            'industry': industry,
            'audience_size': audience_size
        }
        
        with st.spinner("Analyzing the best marketing strategies for your business..."):
            # Get recommendations
            recommendations = recommender.get_recommendations(user_prefs, top_n=3)
            
            if len(recommendations) == 0:
                st.warning("No matching strategies found. Try adjusting your criteria.")
            else:
                st.header("Recommended Marketing Strategies")
                
                # Display each recommendation
                for i, (_, row) in enumerate(recommendations.iterrows()):
                    strategy_id = int(row['strategy_id'])
                    strategy_details = recommender.get_strategy_details(strategy_id)
                    
                    # Create expandable section for each recommendation
                    with st.expander(f"{i+1}. {row['strategy_name']} (Match: {row['similarity_score']:.2f})", expanded=(i==0)):
                        col1, col2 = st.columns([3, 2])
                        
                        with col1:
                            st.subheader("Strategy Overview")
                            st.write(f"**Best Industries:** {strategy_details['best_for_industry']}")
                            
                            # Create rating displays
                            metrics = {
                                "Budget Required": strategy_details['budget_required'],
                                "Technical Expertise": strategy_details['technical_expertise'],
                                "Time Investment": strategy_details['time_investment'],
                                "Conversion Rate Potential": strategy_details['conversion_rate'],
                                "Brand Awareness Impact": strategy_details['brand_awareness'],
                                "Lead Generation Capability": strategy_details['lead_generation'],
                                "Customer Retention Effect": strategy_details['customer_retention']
                            }
                            
                            for metric_name, value in metrics.items():
                                st.write(f"**{metric_name}:** {'⭐' * value} ({value}/5)")
                        
                        with col2:
                            # Create and display radar chart
                            radar_fig = create_radar_chart(strategy_details)
                            st.pyplot(radar_fig)
    
    # Show all strategies table
    with st.expander("View All Available Marketing Strategies"):
        st.dataframe(
            recommender.strategies_df[['strategy_name', 'budget_required', 'technical_expertise', 
                                     'time_investment', 'conversion_rate', 'brand_awareness', 
                                     'lead_generation', 'customer_retention', 'best_for_industry']]
        )
    
    st.markdown("---")
    st.caption("© 2023 Digital Marketing Strategy Recommender | Personalized Marketing Solutions for SMEs")

if __name__ == "__main__":
    main()