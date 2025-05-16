# SlateMate: AI-Based Digital Wellbeing Assistant for Children

## Overview
SlateMate is an AI-powered digital wellbeing assistant designed to help parents understand and support their children's digital habits. Using advanced machine learning and statistical analysis, SlateMate provides personalized insights and recommendations based on screen time patterns, app usage, and emotional responses.

![SlateMate Logo](https://media.licdn.com/dms/image/v2/D4D0BAQEns01azIKK8Q/company-logo_200_200/B4DZXxlh5GHkAI-/0/1743514903145/slate_mate_logo?e=1752710400&v=beta&t=PBMcLYXYFfKosdenuIMCrSjkFbwD_fR24wEPfdBqfqU)

## Features
- **Digital Habit Classification**: Identifies four distinct digital behavior patterns
- **Emotion Detection**: Analyzes journal entries to understand emotional states
- **Personalized Recommendations**: Provides customized guidance for each child
- **Comprehensive Analytics**: Delivers detailed metrics and visualizations

## Project Structure
The project consists of two main components:
1. **Jupyter Notebook** (`AI_Digital_Wellbeing_Assistant_for_Children.ipynb`): Contains data analysis, model development, and visualization code
2. **Streamlit App** (`app.py`): Interactive web application for parents to explore insights

## Digital Habit Types
SlateMate classifies children's digital behavior into four categories:
- **Balanced Explorer** ⚖️: Uses technology in a balanced way across categories with regular breaks
- **Mindless Scroller** 📱: Tends to use social media excessively with few breaks and little purpose
- **Stress Escaper** 🎮: Uses technology to escape negative emotions, often with entertainment apps
- **Goal-Oriented Learner** 📚: Primarily uses educational apps with clear purpose and good break habits

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone https://github.com/sudipto0315/AI_Digital_Wellbeing_Assistant_for_Children.git
cd AI_Digital_Wellbeing_Assistant_for_Children
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv .venv
source .venv/bin/activate  
# On Windows: .venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Jupyter Notebook
To explore the data analysis and model development:
```bash
jupyter notebook AI_Digital_Wellbeing_Assistant_for_Children.ipynb
```

### Running the Streamlit App
To launch the interactive dashboard:
```bash
streamlit run app.py
```

The app will be available at http://localhost:8501 in your web browser.

## Dashboard Features
- **Dashboard Overview**: View overall statistics and trends across all students
- **Individual Reports**: Select specific students to see detailed profiles and recommendations
- **Analysis**: Explore deeper patterns and correlations in the data
- **About**: Learn more about the SlateMate platform and methodology

## Data Format
SlateMate expects input data in CSV format with the following columns:
- `student_id`: Unique identifier for each student
- `screen_time_log`: List of screen time minutes per session
- `trigger_event`: Event that triggered screen usage
- `emotion_journal`: Text entry describing emotional state
- `app_category_usage`: Dictionary of app usage by category
- `parent_warning_ignored`: Boolean indicating if warnings were ignored
- `used_focus_mode`: Boolean indicating focus mode usage
- `screen_breaks_taken`: Number of breaks taken during screen time
- `reward_unlocked`: Boolean indicating if rewards were unlocked

A sample dataset (`slatemate_ai_parents_control_dataset.csv`) is included for demonstration.

## Methodology
SlateMate uses:
- **K-means clustering** for digital habit classification
- **VADER sentiment analysis** for emotion detection
- **Statistical analysis** for screen time patterns
- **Visualization techniques** for intuitive insights

## Privacy & Ethics
All data processing happens locally within the application. No data is sent to external servers. SlateMate is designed to empower parents while respecting children's autonomy and privacy. The focus is on fostering healthy conversations rather than surveillance.

## Requirements
```
pandas==2.2.3
numpy==2.2.5
matplotlib==3.10.3
seaborn==0.13.2
scikit-learn==1.6.1
vaderSentiment==3.3.2
streamlit==1.32.0
plotly==5.18.0
scipy==1.15.3
joblib==1.5.0
threadpoolctl==3.6.0
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- VADER Sentiment Analysis for emotion detection
- Streamlit for the interactive web application
- Plotly for advanced visualizations

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

        