import http.server
import socketserver
import json
import os
import sys
import urllib.parse
from simple_recommender import SimpleMarketingRecommender

# Initialize the recommender
recommender = SimpleMarketingRecommender()

def extract_all_industries():
    """Extract all unique industries from the dataset"""
    all_industries = set()
    for strategy in recommender.strategies:
        industries = strategy['best_for_industry'].replace('"', '').split(',')
        all_industries.update(industries)
    return sorted(list(all_industries))

# Get all industries for the form
all_industries = extract_all_industries()

# HTML template for the main page
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Digital Marketing Strategy Recommender for SMEs</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }
        header {
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 1px solid #ddd;
            padding-bottom: 20px;
        }
        h1 {
            color: #2c3e50;
        }
        .form-container {
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 30px;
        }
        label {
            display: block;
            margin: 15px 0 5px;
            font-weight: bold;
        }
        select, input[type="range"] {
            width: 100%;
            padding: 8px;
            margin-bottom: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .range-container {
            display: flex;
            align-items: center;
        }
        .range-container input {
            flex: 1;
        }
        .range-container output {
            width: 30px;
            text-align: center;
            font-weight: bold;
        }
        button {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            border-radius: 4px;
            cursor: pointer;
            transition: background-color 0.3s;
            margin-top: 15px;
        }
        button:hover {
            background-color: #2980b9;
        }
        .recommendations {
            margin-top: 30px;
        }
        .recommendation {
            background-color: #f9f9f9;
            border-left: 5px solid #3498db;
            padding: 15px;
            margin-bottom: 20px;
        }
        .recommendation h3 {
            margin-top: 0;
            color: #2c3e50;
        }
        .rating {
            font-size: 18px;
            color: #f39c12;
        }
        .similarity {
            color: #27ae60;
            font-weight: bold;
        }
        .requirements, .outcomes {
            margin-top: 15px;
        }
        footer {
            margin-top: 50px;
            text-align: center;
            color: #7f8c8d;
            border-top: 1px solid #ddd;
            padding-top: 20px;
        }
    </style>
</head>
<body>
    <header>
        <h1>Digital Marketing Strategy Recommender for SMEs</h1>
        <p>Find the most suitable digital marketing strategies based on your specific needs, resources, and goals.</p>
    </header>
    
    <div class="form-container">
        <form id="recommenderForm" method="post" action="/recommend">
            <label for="industry">Industry:</label>
            <select id="industry" name="industry" required>
                <option value="" disabled selected>Select your industry</option>
                {industry_options}
            </select>
            
            <label for="budget_amount">Monthly Budget Amount ($):</label>
            <input type="number" id="budget_amount" name="budget_amount" min="0" step="100" value="2000" required>
            
            <label for="daily_hours">Daily Hours Available:</label>
            <input type="number" id="daily_hours" name="daily_hours" min="1" max="24" value="8" required>
            
            <label for="technical">Technical Skills:</label>
            <div class="range-container">
                <input type="range" id="technical" name="technical_skill" min="1" max="5" value="3" oninput="this.nextElementSibling.value = this.value">
                <output>3</output>
                <span>&nbsp;(1=Beginner, 5=Expert)</span>
            </div>
            
            <label for="audience">Target Audience Size:</label>
            <div class="range-container">
                <input type="range" id="audience" name="audience_size" min="1" max="5" value="3" oninput="this.nextElementSibling.value = this.value">
                <output>3</output>
                <span>&nbsp;(1=Very Small, 5=Very Large)</span>
            </div>
            
            <label for="goal">Primary Marketing Goal:</label>
            <select id="goal" name="goal" required>
                <option value="conversion">Increase Conversion Rates</option>
                <option value="awareness">Boost Brand Awareness</option>
                <option value="leads">Generate More Leads</option>
                <option value="retention">Improve Customer Retention</option>
            </select>
            
            <button type="submit">Generate Recommendations</button>
        </form>
    </div>
    
    {recommendations_html}
    
    <footer>
        <p>&copy; 2023 Digital Marketing Strategy Recommender | Personalized Marketing Solutions for SMEs</p>
    </footer>

</body>
</html>
'''

# HTML template for recommendations
RECOMMENDATIONS_TEMPLATE = '''
<div class="recommendations">
    <h2>Recommended Marketing Strategies</h2>
    {recommendation_items}
</div>
'''

# HTML template for a single recommendation
RECOMMENDATION_ITEM_TEMPLATE = '''
<div class="recommendation">
    <h3>{strategy_name}</h3>
    <p><span class="similarity">Match Score: {similarity:.2f}</span></p>
    <p><strong>Best for:</strong> {best_for}</p>
    
    <div class="requirements">
        <h4>Resource Requirements:</h4>
        <p><strong>Budget Required:</strong> <span class="rating">{"$" * budget_required}</span> ({budget_required}/5)</p>
        <p><strong>Technical Expertise:</strong> <span class="rating">{"*" * technical_expertise}</span> ({technical_expertise}/5)</p>
        <p><strong>Time Investment:</strong> <span class="rating">{"⏱" * time_investment}</span> ({time_investment}/5)</p>
    </div>
    
    <div class="outcomes">
        <h4>Expected Outcomes:</h4>
        <p><strong>Conversion Rate:</strong> <span class="rating">{"↑" * conversion_rate}</span> ({conversion_rate}/5)</p>
        <p><strong>Brand Awareness:</strong> <span class="rating">{"👁" * brand_awareness}</span> ({brand_awareness}/5)</p>
        <p><strong>Lead Generation:</strong> <span class="rating">{"⚡" * lead_generation}</span> ({lead_generation}/5)</p>
        <p><strong>Customer Retention:</strong> <span class="rating">{"♥" * customer_retention}</span> ({customer_retention}/5)</p>
    </div>
</div>
'''

class RecommenderHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            # Generate industry options
            industry_options = ""
            for industry in all_industries:
                industry_options += f'<option value="{industry}">{industry}</option>\n'
            
            # Render HTML template
            html = HTML_TEMPLATE.format(
                industry_options=industry_options,
                recommendations_html=""
            )
            
            self.wfile.write(html.encode())
        else:
            # Serve static files
            super().do_GET()
    
    def do_POST(self):
        """Handle POST requests"""
        if self.path == '/recommend':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode('utf-8')
            form_data = urllib.parse.parse_qs(post_data)
            
            # Extract form values
            user_prefs = {
                'industry': form_data.get('industry', [''])[0],
                'budget_amount': int(form_data.get('budget_amount', ['2000'])[0]),
                'daily_hours': int(form_data.get('daily_hours', ['8'])[0]),
                'technical_skill': int(form_data.get('technical_skill', ['3'])[0]),
                'audience_size': int(form_data.get('audience_size', ['3'])[0]),
                'goal': form_data.get('goal', ['awareness'])[0]
            }
            
            # Get recommendations
            recommendations = recommender.get_recommendations(user_prefs, top_n=3)
            
            # Generate recommendations HTML
            recommendation_items = ""
            for strategy, similarity in recommendations:
                recommendation_items += RECOMMENDATION_ITEM_TEMPLATE.format(
                    strategy_name=strategy['strategy_name'],
                    similarity=similarity,
                    best_for=strategy['best_for_industry'],
                    budget_required=strategy['budget_required'],
                    technical_expertise=strategy['technical_expertise'],
                    time_investment=strategy['time_investment'],
                    conversion_rate=strategy['conversion_rate'],
                    brand_awareness=strategy['brand_awareness'],
                    lead_generation=strategy['lead_generation'],
                    customer_retention=strategy['customer_retention']
                )
            
            recommendations_html = RECOMMENDATIONS_TEMPLATE.format(
                recommendation_items=recommendation_items
            ) if recommendations else ""
            
            # Generate industry options
            industry_options = ""
            for industry in all_industries:
                selected = 'selected' if industry == user_prefs['industry'] else ''
                industry_options += f'<option value="{industry}" {selected}>{industry}</option>\n'
            
            # Render HTML response
            html = HTML_TEMPLATE.format(
                industry_options=industry_options,
                recommendations_html=recommendations_html
            )
            
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(html.encode())
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not Found')

def run_server(port=8000):
    """Run the web server on the specified port"""
    print(f"Starting server at http://localhost:{port}")
    httpd = socketserver.TCPServer(("", port), RecommenderHandler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        httpd.server_close()

if __name__ == "__main__":
    # Check if recommender was initialized successfully
    if not recommender.strategies:
        print("Failed to load marketing strategies data. Exiting...")
        sys.exit(1)
    
    # Use port 8000 by default or from command line
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port: {sys.argv[1]}. Using default port 8000.")
    
    run_server(port)