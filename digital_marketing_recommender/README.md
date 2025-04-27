# Digital Marketing Strategy Recommender for SMEs

A personalized recommendation system that helps small and medium enterprises (SMEs) identify the most suitable digital marketing strategies based on their specific needs, resources, and goals.

## Features

- **Personalized Recommendations**: Get tailored marketing strategy recommendations based on your business profile
- **Zero Dependencies**: The simple version runs with standard Python libraries only
- **Multiple Interfaces**: Choose between command-line, GUI, or run simple tests
- **Industry-Specific**: Filter strategies that work best for your industry
- **Resource-Aware**: Considers your budget, technical expertise, and available time
- **Goal-Oriented**: Focuses on your primary marketing objective (conversion, awareness, leads, retention)
- **NEW!** Accessible futuristic interface with improved layout and visibility

## Getting Started

### Prerequisites

- Python 3.6 or higher
- Tkinter (included with most Python installations)

### Installation

No additional packages are required. Simply clone or download this repository to your local machine.

## Usage

### Command-Line Interface

Run the CLI for an interactive text-based experience:
```
cd digital_marketing_recommender
python marketing_recommender_cli.py
```

Follow the interactive prompts to input your business profile.

### Graphical User Interface (GUI)

For a more user-friendly experience, run the GUI version:
```
cd digital_marketing_recommender
python gui_recommender.py
```

This provides a simple form with sliders and dropdown menus to enter your business profile and view recommendations.

### Futuristic GUI Interface

For a modern interface with enhanced visuals:
```
cd digital_marketing_recommender
python futuristic_gui.py
```

### NEW! Accessible Futuristic Interface

For an improved interface with better layout and accessibility features:
```
cd digital_marketing_recommender
python accessible_futuristic_gui.py
```

Key improvements:
- Tabbed interface for better organization
- Larger window size for improved visibility
- Radio buttons instead of sliders for clearer selections
- Integrated help system with tooltips on all controls
- Enhanced status indicators

### Simple Test

For a quick demonstration with preset values:
```
cd digital_marketing_recommender
python simple_recommender.py
```

## Interface Options

1. **GUI Interface** (`gui_recommender.py`) - Recommended for most users
   - Easy-to-use graphical interface
   - Sliders for numerical inputs
   - Results displayed in scrollable text area
   - No additional dependencies (uses standard tkinter)

2. **Command-Line Interface** (`marketing_recommender_cli.py`)
   - Text-based interactive experience
   - Works in any terminal or command prompt
   - Simple navigation with numbered options
   - Clear, formatted output

3. **Simple Test** (`simple_recommender.py`)
   - Quick demonstration with preset values
   - Basic output for testing and verification
   - Example implementation for developers

## How It Works

The recommendation system uses a content-based filtering approach with cosine similarity:

1. **Data Collection**: A curated dataset of digital marketing strategies with their attributes
2. **User Profile**: Your inputs create a numerical profile representing your business needs
3. **Similarity Calculation**: Cosine similarity between your profile and marketing strategies
4. **Filtering & Ranking**: Strategies are filtered by industry compatibility and ranked by similarity score

## Project Structure

```
digital_marketing_recommender/
├── data/
│   └── marketing_strategies.csv    # Dataset of marketing strategies
├── simple_recommender.py           # Core recommendation engine
├── marketing_recommender_cli.py    # Command-line interface
├── gui_recommender.py              # Graphical user interface (GUI)
├── futuristic_gui.py               # Futuristic GUI interface
├── accessible_futuristic_gui.py    # Accessible futuristic GUI interface
└── README.md                       # Documentation
```

## Customization

You can customize the recommendation system by:

1. **Adding New Strategies**: Edit the `marketing_strategies.csv` file to add more digital marketing strategies
2. **Adjusting the Algorithm**: Modify the feature weights in the `get_recommendations` method in `simple_recommender.py`
3. **Enhancing the UI**: Customize the GUI in `gui_recommender.py` or the CLI in `marketing_recommender_cli.py`

## Future Improvements

Potential enhancements for future versions:
- Web-based interface for access from any device
- Additional strategy attributes and filtering options
- Integration with real-world marketing data
- Machine learning to improve recommendations over time