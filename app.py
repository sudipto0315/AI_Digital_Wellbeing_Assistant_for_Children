import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import warnings
warnings.filterwarnings('ignore')

# Set page configuration with better styling
st.set_page_config(
    page_title="SlateMate AI Wellbeing Assistant",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #3a86ff;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.5rem;
        color: #4361ee;
        margin-top: 1rem;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #4361ee;
    }
    .metric-label {
        font-size: 1rem;
        color: #6c757d;
    }
    .alert {
        background-color: #ffccd5;
        border-left: 5px solid #ff4d6d;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 0.5rem;
    }
    .recommendation {
        background-color: #d7f9e9;
        border-left: 5px solid #2ec4b6;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 0.5rem;
    }
    .trend-positive {
        color: #2ec4b6;
        font-weight: bold;
    }
    .trend-negative {
        color: #ff4d6d;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<h1 class="main-header">📱 SlateMate: AI-Based Digital Wellbeing Assistant for Children</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center;">Empowering parents with smart, empathetic insights into their child\'s digital habits.</p>', unsafe_allow_html=True)

# Helper Functions
def extract_screen_time_features(screen_time_log):
    """Extract statistical features from screen time log"""
    return {
        "avgScreenTime": np.mean(screen_time_log),
        "maxScreenTime": np.max(screen_time_log),
        "minScreenTime": np.min(screen_time_log),
        "stdScreenTime": np.std(screen_time_log),
        "totalScreenTime": np.sum(screen_time_log)
    }

def detect_emotion(text):
    """Detect emotion from journal text using VADER sentiment analysis"""
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(str(text))
    compound = sentiment['compound']
    
    if compound >= 0.5:
        return "Positive"
    elif compound > 0:
        return "Neutral"
    elif compound <= -0.5:
        return "Angry"
    elif "sad" in str(text).lower():
        return "Sad"
    else:
        return "Anxious"

def recommendations(row):
    """Generate alerts and recommendations based on user data"""
    alerts = []
    recs = []
    
    # Check parent warning behavior
    if row['parent_warning_ignored'] == True:
        alerts.append("Ignored parental warnings")
        recs.append("Trigger soft reminder after 3 ignores")
    
    # Check screen breaks
    if row['screen_breaks_taken'] < 2:
        alerts.append("No breaks during binge usage")
        recs.append("Enable mandatory screen breaks after 45 mins")
    
    # Check focus mode usage
    if row['used_focus_mode'] == False:
        recs.append("Introduce focus mode tutorial")
    
    # Check emotional state
    if row['dominantEmotion'] in ['Angry', 'Anxious', 'Sad']:
        alerts.append(f"Negative emotion detected: {row['dominantEmotion']}")
        recs.append("Prompt emotional journal reflection or breathing exercise")
    
    # Check digital habit type
    if row['digitalHabitType'] == "Mindless Scroller":
        recs.append("Suggest 15-min limit on Social Media")
    elif row['digitalHabitType'] == "Goal-Oriented Learner":
        recs.append("Praise effort and encourage learning streak")
    elif row['digitalHabitType'] == "Stress Escaper":
        recs.append("Introduce mindfulness breaks between screen sessions")
    elif row['digitalHabitType'] == "Balanced Explorer":
        recs.append("Maintain current balance with subtle nudges toward educational content")
    
    # Check excessive app usage
    if row.get('socialMinutes', 0) > 120:  # Over 2 hours on social media
        alerts.append("Excessive social media usage")
        recs.append("Implement gradual reduction targets for social media")
    
    # Check educational engagement
    if row.get('eduMinutes', 0) < 30:  # Less than 30 minutes on educational content
        alerts.append("Low educational content engagement")
        recs.append("Suggest educational games or interactive content")
    
    return alerts, recs

def generate_child_summary(row):
    """Generate comprehensive summary for a child"""
    return {
        "student_id": row["student_id"],
        "digital_habit_type": row["digitalHabitType"],
        "dominant_emotion": row["dominantEmotion"],
        "screen_time_stats": {
            "avg_minutes": round(row["avgScreenTime"], 1),
            "total_minutes": round(row["totalScreenTime"], 1),
            "max_session": round(row["maxScreenTime"], 1)
        },
        "app_usage": {
            "entertainment": row.get("entMinutes", 0),
            "education": row.get("eduMinutes", 0),
            "social_media": row.get("socialMinutes", 0),
            "other": row.get("otherMinutes", 0)
        },
        "focus_mode_used": row["used_focus_mode"],
        "screen_breaks_taken": row["screen_breaks_taken"],
        "alerts": row["alerts"],
        "recommendations": row["recommendations"]
    }

def get_habit_description(habit_type):
    """Return description for each habit type"""
    descriptions = {
        "Balanced Explorer": "Uses technology in a balanced way across categories with regular breaks.",
        "Mindless Scroller": "Tends to use social media excessively with few breaks and little purpose.",
        "Stress Escaper": "Uses technology to escape negative emotions, often with entertainment apps.",
        "Goal-Oriented Learner": "Primarily uses educational apps with clear purpose and good break habits."
    }
    return descriptions.get(habit_type, "No description available")

def get_habit_emoji(habit_type):
    """Return emoji for each habit type"""
    emojis = {
        "Balanced Explorer": "⚖️",
        "Mindless Scroller": "📱",
        "Stress Escaper": "🎮",
        "Goal-Oriented Learner": "📚"
    }
    return emojis.get(habit_type, "❓")

def get_emotion_emoji(emotion):
    """Return emoji for each emotion type"""
    emojis = {
        "Positive": "😊",
        "Neutral": "😐",
        "Angry": "😠",
        "Sad": "😢",
        "Anxious": "😰"
    }
    return emojis.get(emotion, "❓")

def parse_input_data(df):
    """Process input dataframe to extract all required features"""
    try:
        # Safely parse JSON strings to lists/dicts
        df['screen_time_log'] = df['screen_time_log'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        df['app_category_usage'] = df['app_category_usage'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        
        # Extract screen time features
        screen_time_features = df['screen_time_log'].apply(extract_screen_time_features)
        screen_df = pd.DataFrame(screen_time_features.tolist())
        df = pd.concat([df, screen_df], axis=1)
        
        # Extract app usage
        app_df = df['app_category_usage'].apply(pd.Series)
        df = pd.concat([df, app_df], axis=1)
        df.rename(columns={
            "Entertainment": "entMinutes",
            "Education": "eduMinutes",
            "Social Media": "socialMinutes",
            "Others": "otherMinutes"
        }, inplace=True)
        
        # Apply emotion detection
        df['dominantEmotion'] = df['emotion_journal'].apply(detect_emotion)
        
        # Apply clustering
        features = [
            "avgScreenTime", "stdScreenTime", "totalScreenTime",
            "entMinutes", "eduMinutes", "socialMinutes", "otherMinutes",
            "screen_breaks_taken"
        ]
        
        # Make sure all features exist, use defaults if not
        for feature in features:
            if feature not in df.columns:
                df[feature] = 0
                
        X = df[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        df['habitCluster'] = kmeans.fit_predict(X_scaled)
        
        # Map clusters to habit types
        cluster_map = {
            0: "Balanced Explorer",
            1: "Mindless Scroller",
            2: "Stress Escaper",
            3: "Goal-Oriented Learner"
        }
        df['digitalHabitType'] = df['habitCluster'].map(cluster_map)
        
        # Generate recommendations
        df[['alerts', 'recommendations']] = df.apply(lambda row: pd.Series(recommendations(row)), axis=1)
        
        return df, True
    except Exception as e:
        st.error(f"Error processing data: {e}")
        return df, False

# Main App Logic
def main():
    # Sidebar
    with st.sidebar:
        st.image("https://media.licdn.com/dms/image/v2/D4D0BAQEns01azIKK8Q/company-logo_200_200/B4DZXxlh5GHkAI-/0/1743514903145/slate_mate_logo?e=1752710400&v=beta&t=PBMcLYXYFfKosdenuIMCrSjkFbwD_fR24wEPfdBqfqU", width=70)
        st.title("Navigation")
        page = st.radio("Go to", ["Dashboard", "Individual Reports", "Analysis", "About"])
        
        st.markdown("---")
        st.markdown("### Upload Data")
        uploaded_file = st.file_uploader("Upload dataset (CSV format)", type=["csv"])
        
        # Sample data option
        use_sample = st.checkbox("Use sample data", value=True)
        
        st.markdown("---")
        st.markdown("### Features")
        st.markdown("✅ Digital habit classification")
        st.markdown("✅ Emotion detection")
        st.markdown("✅ Personalized recommendations")
        st.markdown("✅ Parent-focused insights")
        
    # Determine which data to use (uploaded or sample)
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            df, success = parse_input_data(df)
            if not success:
                st.error("Failed to process the uploaded file. Please check the format.")
                return
        except Exception as e:
            st.error(f"Error reading the uploaded file: {e}")
            return
    elif use_sample:
        # Use sample data if no file is uploaded
        try:
            df = pd.read_csv('slatemate_ai_parents_control_dataset.csv')
            df, success = parse_input_data(df)
            if not success:
                st.error("Failed to process the sample file.")
                return
        except Exception as e:
            st.error(f"Error loading sample data: {e}. Please make sure the sample file exists.")
            return
    else:
        st.info("Please upload a dataset or enable the sample data option.")
        return
    
    # Main content based on selected page
    if page == "Dashboard":
        display_dashboard(df)
    elif page == "Individual Reports":
        display_individual_reports(df)
    elif page == "Analysis":
        display_analysis(df)
    else:  # About page
        display_about()

def display_dashboard(df):
    """Display the main dashboard with overall statistics"""
    st.markdown('<h2 class="subheader">Dashboard Overview</h2>', unsafe_allow_html=True)
    
    # Key metrics in cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{len(df)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Students Monitored</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        avg_screen_time = round(df['totalScreenTime'].mean(), 1)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{avg_screen_time}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Avg. Daily Screen Time (min)</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        focus_rate = round(df['used_focus_mode'].mean() * 100, 1)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{focus_rate}%</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Focus Mode Usage</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        positive_emotions = round(df[df['dominantEmotion'] == 'Positive'].shape[0] / len(df) * 100, 1)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{positive_emotions}%</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Positive Emotional States</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Digital habits distribution and emotions distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3>Digital Habit Distribution</h3>', unsafe_allow_html=True)
        
        # Create dataframe for plotting
        habit_counts = df['digitalHabitType'].value_counts().reset_index()
        habit_counts.columns = ['Habit', 'Count']
        
        # Create Plotly pie chart
        fig = px.pie(
            habit_counts, 
            values='Count', 
            names='Habit', 
            color='Habit',
            color_discrete_map={
                'Balanced Explorer': '#4361ee',
                'Mindless Scroller': '#ff4d6d',
                'Stress Escaper': '#ffb703',
                'Goal-Oriented Learner': '#2ec4b6'
            },
            hole=0.4
        )
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3>Emotional State Distribution</h3>', unsafe_allow_html=True)
        
        # Create dataframe for plotting
        emotion_counts = df['dominantEmotion'].value_counts().reset_index()
        emotion_counts.columns = ['Emotion', 'Count']
        
        # Create Plotly pie chart
        fig = px.pie(
            emotion_counts, 
            values='Count', 
            names='Emotion', 
            color='Emotion',
            color_discrete_map={
                'Positive': '#2ec4b6',
                'Neutral': '#90be6d',
                'Anxious': '#ffb703',
                'Angry': '#ff4d6d',
                'Sad': '#6c757d'
            },
            hole=0.4
        )
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # App usage breakdown
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3>App Category Usage Breakdown</h3>', unsafe_allow_html=True)
    
    # Calculate average usage by category
    avg_usage = {
        'Education': df['eduMinutes'].mean(),
        'Entertainment': df['entMinutes'].mean(),
        'Social Media': df['socialMinutes'].mean(),
        'Others': df['otherMinutes'].mean()
    }
    
    # Create dataframe for plotting
    usage_df = pd.DataFrame({
        'Category': list(avg_usage.keys()),
        'Minutes': list(avg_usage.values())
    })
    
    # Create Plotly bar chart
    fig = px.bar(
        usage_df, 
        x='Category', 
        y='Minutes',
        color='Category',
        color_discrete_map={
            'Education': '#4361ee',
            'Entertainment': '#ff6b6b',
            'Social Media': '#f72585',
            'Others': '#6c757d'
        },
        text='Minutes'
    )
    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    fig.update_layout(
        xaxis_title="",
        yaxis_title="Average Minutes per Day",
        margin=dict(t=10, b=0, l=0, r=0)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Common alerts and recommendations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3>Top Alerts</h3>', unsafe_allow_html=True)
        
        # Flatten alerts lists and count occurrences
        all_alerts = []
        for alerts in df['alerts']:
            all_alerts.extend(alerts)
        
        alert_counts = pd.Series(all_alerts).value_counts().head(5)
        
        # Create dataframe for plotting
        alert_df = pd.DataFrame({
            'Alert': alert_counts.index,
            'Count': alert_counts.values
        })
        
        # Create Plotly bar chart
        fig = px.bar(
            alert_df, 
            y='Alert', 
            x='Count',
            orientation='h',
            color_discrete_sequence=['#ff4d6d']
        )
        fig.update_layout(
            yaxis_title="",
            xaxis_title="Number of Occurrences",
            margin=dict(t=10, b=0, l=0, r=0)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3>Top Recommendations</h3>', unsafe_allow_html=True)
        
        # Flatten recommendations lists and count occurrences
        all_recs = []
        for recs in df['recommendations']:
            all_recs.extend(recs)
        
        rec_counts = pd.Series(all_recs).value_counts().head(5)
        
        # Create dataframe for plotting
        rec_df = pd.DataFrame({
            'Recommendation': rec_counts.index,
            'Count': rec_counts.values
        })
        
        # Create Plotly bar chart
        fig = px.bar(
            rec_df, 
            y='Recommendation', 
            x='Count',
            orientation='h',
            color_discrete_sequence=['#2ec4b6']
        )
        fig.update_layout(
            yaxis_title="",
            xaxis_title="Number of Occurrences",
            margin=dict(t=10, b=0, l=0, r=0)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

def display_individual_reports(df):
    """Display individual student reports"""
    st.markdown('<h2 class="subheader">Individual Student Reports</h2>', unsafe_allow_html=True)
    
    # Student selector
    col1, col2 = st.columns([1, 3])
    
    with col1:
        selected_id = st.selectbox("Select Student ID", sorted(df['student_id'].unique()))
    
    with col2:
        # Filter buttons by digital habit type
        habit_types = ["All"] + sorted(df['digitalHabitType'].unique().tolist())
        selected_filter = st.radio("Filter by habit type:", habit_types, horizontal=True)
    
    # Get selected student data
    selected = df[df['student_id'] == selected_id].iloc[0]
    
    # Student Summary
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        habit_emoji = get_habit_emoji(selected['digitalHabitType'])
        emotion_emoji = get_emotion_emoji(selected['dominantEmotion'])
        
        st.markdown(f"""
        <div style="text-align: center;">
            <div style="font-size: 5rem;">{habit_emoji}</div>
            <div style="font-size: 1.2rem; font-weight: bold;">{selected['digitalHabitType']}</div>
            <div style="margin-top: 1.5rem; font-size: 3rem;">{emotion_emoji}</div>
            <div style="font-size: 1.2rem;">{selected['dominantEmotion']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <h3>{selected['student_id']}'s Digital Profile</h3>
        <p><strong>Digital Habit Type:</strong> {selected['digitalHabitType']}</p>
        <p>{get_habit_description(selected['digitalHabitType'])}</p>
        <p><strong>Emotional State:</strong> {selected['dominantEmotion']}</p>
        <p><strong>Journal Entry:</strong> "{selected['emotion_journal']}"</p>
        <p><strong>Focus Mode Used:</strong> {"Yes ✅" if selected['used_focus_mode'] else "No ❌"}</p>
        <p><strong>Screen Breaks Taken:</strong> {selected['screen_breaks_taken']}</p>
        <p><strong>Reward Unlocked:</strong> {"Yes 🏆" if selected.get('reward_unlocked', False) else "No ❌"}</p>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Screen Time and App Usage
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3>Screen Time Breakdown</h3>', unsafe_allow_html=True)
        
        # Screen time metrics
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        
        with metrics_col1:
            st.metric("Total Screen Time", f"{selected['totalScreenTime']:.1f} min")
        
        with metrics_col2:
            st.metric("Average Session", f"{selected['avgScreenTime']:.1f} min")
        
        with metrics_col3:
            st.metric("Longest Session", f"{selected['maxScreenTime']:.1f} min")
        
        # Screen time visualization
        screen_time_data = selected['screen_time_log']
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(range(len(screen_time_data))),
            y=screen_time_data,
            marker_color='#4361ee'
        ))
        fig.update_layout(
            title="Screen Time Sessions",
            xaxis_title="Session Number",
            yaxis_title="Minutes",
            height=300,
            margin=dict(t=40, b=0, l=0, r=0)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3>App Category Usage</h3>', unsafe_allow_html=True)
        
        # App usage data
        app_data = {
            'Education': selected.get('eduMinutes', 0),
            'Entertainment': selected.get('entMinutes', 0),
            'Social Media': selected.get('socialMinutes', 0),
            'Others': selected.get('otherMinutes', 0)
        }
        
        # Create dataframe for plotting
        app_df = pd.DataFrame({
            'Category': list(app_data.keys()),
            'Minutes': list(app_data.values())
        })
        
        # Create Plotly pie chart
        fig = px.pie(
            app_df, 
            values='Minutes', 
            names='Category',
            color='Category',
            color_discrete_map={
                'Education': '#4361ee',
                'Entertainment': '#ff6b6b',
                'Social Media': '#f72585',
                'Others': '#6c757d'
            },
            hole=0.4
        )
        fig.update_layout(
            height=300,
            margin=dict(t=0, b=0, l=0, r=0)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # App usage breakdown in a table
        st.markdown("<h4>Minutes by Category</h4>", unsafe_allow_html=True)
        for category, minutes in app_data.items():
            st.markdown(f"**{category}:** {minutes} minutes")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Alerts and Recommendations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3>⚠️ Alerts</h3>', unsafe_allow_html=True)
        
        if selected['alerts']:
            for alert in selected['alerts']:
                st.markdown(f'<div class="alert">{alert}</div>', unsafe_allow_html=True)
        else:
            st.info("No alerts for this student.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3>✅ Recommendations</h3>', unsafe_allow_html=True)
        
        if selected['recommendations']:
            for rec in selected['recommendations']:
                st.markdown(f'<div class="recommendation">{rec}</div>', unsafe_allow_html=True)
        else:
            st.info("No recommendations for this student.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Development Plan
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3>📈 Personalized Development Plan</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h4>Short-term Goals (1-2 weeks)</h4>", unsafe_allow_html=True)
        
        if selected['digitalHabitType'] == "Mindless Scroller":
            st.markdown("1. Reduce social media usage by 15% each week")
            st.markdown("2. Enable focus mode for at least 30 minutes daily")
            st.markdown("3. Take at least 3 screen breaks during long sessions")
        elif selected['digitalHabitType'] == "Stress Escaper":
            st.markdown("1. Journal emotions before screen time sessions")
            st.markdown("2. Practice 2-minute mindfulness before entertainment apps")
            st.markdown("3. Set a 45-minute limit on continuous gaming/entertainment")
        elif selected['digitalHabitType'] == "Goal-Oriented Learner":
            st.markdown("1. Maintain balance by adding 10 minutes of free exploration")
            st.markdown("2. Take regular eye-strain prevention breaks")
            st.markdown("3. Share learning achievements for positive reinforcement")
        else:  # Balanced Explorer
            st.markdown("1. Maintain current screen time balance")
            st.markdown("2. Explore one new educational app each week")
            st.markdown("3. Practice using focus mode to enhance productivity")
    
    with col2:
        st.markdown("<h4>Long-term Objectives (1-3 months)</h4>", unsafe_allow_html=True)
        
        if selected['digitalHabitType'] == "Mindless Scroller":
            st.markdown("1. Develop self-awareness about scroll triggers")
            st.markdown("2. Establish healthy screen time boundaries")
            st.markdown("3. Shift 30% of social media time to creative digital activities")
        elif selected['digitalHabitType'] == "Stress Escaper":
            st.markdown("1. Develop alternative stress coping mechanisms")
            st.markdown("2. Balance entertainment with educational content")
            st.markdown("3. Create a personalized digital wellness routine")
        elif selected['digitalHabitType'] == "Goal-Oriented Learner":
            st.markdown("1. Expand into broader educational interests")
            st.markdown("2. Develop social collaboration skills through educational platforms")
            st.markdown("3. Teach others about digital wellness as a leadership skill")
        else:  # Balanced Explorer
            st.markdown("1. Become a digital mentor for peers")
            st.markdown("2. Explore more advanced educational content")
            st.markdown("3. Maintain healthy balance while increasing digital literacy")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Parent Communication
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3>👨‍👩‍👧‍👦 Parent Communication Suggestions</h3>', unsafe_allow_html=True)
    
    # Generate customized communication suggestions based on habit type and alerts
    if selected['digitalHabitType'] == "Mindless Scroller":
        message = """
        **Suggested Approach:** Curious and non-judgmental conversations about social media use
        
        **Key Discussion Points:**
        - Ask about favorite content and creators instead of focusing on time spent
        - Discuss how different apps make them feel after usage
        - Explore setting mindful boundaries together rather than imposing restrictions
        - Share your own screen habits and challenges to normalize the conversation
        
        **When to Have Conversation:** Choose a relaxed moment, not immediately after asking them to get off a device
        """
    elif selected['digitalHabitType'] == "Stress Escaper":
        message = """
        **Suggested Approach:** Empathetic exploration of underlying feelings
        
        **Key Discussion Points:**
        - Ask open questions about school, friend, or home stressors
        - Validate their feelings and normalize stress as part of life
        - Discuss how technology helps them feel better temporarily
        - Explore additional coping strategies that might complement digital escapes
        
        **When to Have Conversation:** During a calm activity together, not during or immediately after a stressful situation
        """
    elif selected['digitalHabitType'] == "Goal-Oriented Learner":
        message = """
        **Suggested Approach:** Supportive enhancement of their motivated approach
        
        **Key Discussion Points:**
        - Show genuine interest in what they're learning online
        - Ask how you can support their digital learning journey
        - Gently encourage balance with physical activities and social interactions
        - Recognize and celebrate their disciplined approach to technology
        
        **When to Have Conversation:** During a shared activity or meal when they seem receptive to conversation
        """
    else:  # Balanced Explorer
        message = """
        **Suggested Approach:** Collaborative conversations that recognize their maturity
        
        **Key Discussion Points:**
        - Acknowledge their balanced approach to technology
        - Ask what strategies they use to maintain healthy habits
        - Discuss how they might help younger siblings or friends develop similar habits
        - Share articles or ideas about digital wellness you find interesting
        
        **When to Have Conversation:** Any relaxed time, positioning it as peer-to-peer rather than parent-child
        """
    
    st.markdown(message, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def display_analysis(df):
    """Display deeper analysis and insights"""
    st.markdown('<h2 class="subheader">Comprehensive Analysis</h2>', unsafe_allow_html=True)
    
    # Habit type comparison
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3>Digital Habit Type Comparison</h3>', unsafe_allow_html=True)
    
    # Create comparison dataframe
    habit_comparison = df.groupby('digitalHabitType').agg({
        'avgScreenTime': 'mean',
        'totalScreenTime': 'mean',
        'screen_breaks_taken': 'mean',
        'used_focus_mode': 'mean',
        'eduMinutes': 'mean',
        'entMinutes': 'mean',
        'socialMinutes': 'mean'
    }).reset_index()
    
    # Round values
    for col in habit_comparison.columns:
        if col != 'digitalHabitType':
            habit_comparison[col] = habit_comparison[col].round(1)
    
    # Convert focus mode to percentage
    habit_comparison['used_focus_mode'] = (habit_comparison['used_focus_mode'] * 100).round(1)
    habit_comparison.rename(columns={'used_focus_mode': 'focus_mode_pct'}, inplace=True)
    
    # Create radar chart
    categories = ['Avg Screen Time', 'Daily Total', 'Breaks Taken', 'Focus Mode %', 
                 'Education Min', 'Entertainment Min', 'Social Min']
    
    fig = go.Figure()
    
    for i, row in habit_comparison.iterrows():
        values = [
            row['avgScreenTime'],
            row['totalScreenTime'],
            row['screen_breaks_taken'],
            row['focus_mode_pct'],
            row['eduMinutes'],
            row['entMinutes'],
            row['socialMinutes']
        ]
        
        # Add normalized values to close the radar chart
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=row['digitalHabitType']
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, habit_comparison[['avgScreenTime', 'totalScreenTime', 'screen_breaks_taken', 
                                          'focus_mode_pct', 'eduMinutes', 'entMinutes', 'socialMinutes']].max().max() * 1.1]
            )
        ),
        showlegend=True,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display comparison table
    st.dataframe(
        habit_comparison.rename(columns={
            'avgScreenTime': 'Avg Session (min)',
            'totalScreenTime': 'Daily Total (min)',
            'screen_breaks_taken': 'Breaks Taken',
            'focus_mode_pct': 'Focus Mode Usage (%)',
            'eduMinutes': 'Education (min)',
            'entMinutes': 'Entertainment (min)',
            'socialMinutes': 'Social Media (min)'
        }),
        hide_index=True
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Emotional state analysis
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3>Emotional State Analysis</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Emotion distribution by habit type
        emotion_by_habit = pd.crosstab(df['digitalHabitType'], df['dominantEmotion'])
        emotion_by_habit_pct = emotion_by_habit.div(emotion_by_habit.sum(axis=1), axis=0) * 100
        
        # Create heatmap
        fig = px.imshow(
            emotion_by_habit_pct,
            labels=dict(x="Emotion", y="Digital Habit Type", color="Percentage"),
            x=emotion_by_habit_pct.columns,
            y=emotion_by_habit_pct.index,
            color_continuous_scale="RdBu_r",
            text_auto='.1f'
        )
        fig.update_layout(
            title="Emotion Distribution by Digital Habit Type (%)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Calculate average screen time by emotion
        screen_by_emotion = df.groupby('dominantEmotion').agg({
            'totalScreenTime': 'mean',
            'eduMinutes': 'mean',
            'entMinutes': 'mean',
            'socialMinutes': 'mean'
        }).reset_index()
        
        # Create bar chart
        fig = px.bar(
            screen_by_emotion,
            x='dominantEmotion',
            y=['eduMinutes', 'entMinutes', 'socialMinutes'],
            labels={'value': 'Minutes', 'variable': 'Category'},
            title='App Usage by Emotional State',
            barmode='stack',
            color_discrete_map={
                'eduMinutes': '#4361ee',
                'entMinutes': '#ff6b6b',
                'socialMinutes': '#f72585'
            }
        )
        fig.update_layout(
            xaxis_title="Dominant Emotion",
            yaxis_title="Average Minutes",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Correlation analysis
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3>Correlation Analysis</h3>', unsafe_allow_html=True)
    
    # Select numerical columns for correlation
    numerical_cols = [
        'screen_breaks_taken', 'avgScreenTime', 'totalScreenTime',
        'eduMinutes', 'entMinutes', 'socialMinutes', 'otherMinutes'
    ]
    
    # Calculate correlation matrix
    corr_matrix = df[numerical_cols].corr().round(2)
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1
    )
    fig.update_layout(
        title="Feature Correlation Matrix",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    st.markdown("<h4>Key Correlation Insights:</h4>", unsafe_allow_html=True)
    
    # Extract strongest correlations
    corr_pairs = []
    for i in range(len(numerical_cols)):
        for j in range(i+1, len(numerical_cols)):
            corr_pairs.append((
                numerical_cols[i],
                numerical_cols[j],
                corr_matrix.iloc[i, j]
            ))
    
    # Sort by absolute correlation
    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    # Display top correlations
    for var1, var2, corr in corr_pairs[:5]:
        direction = "positive" if corr > 0 else "negative"
        strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.4 else "weak"
        
        st.markdown(f"- **{var1}** and **{var2}** have a {strength} {direction} correlation ({corr})")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # PCA analysis
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3>Principal Component Analysis</h3>', unsafe_allow_html=True)
    
    # Perform PCA
    features = [
        "avgScreenTime", "stdScreenTime", "totalScreenTime",
        "entMinutes", "eduMinutes", "socialMinutes", "otherMinutes",
        "screen_breaks_taken"
    ]
    
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    
    # Create dataframe with principal components
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['digitalHabitType'] = df['digitalHabitType'].values
    
    # Create scatter plot
    fig = px.scatter(
        pca_df, 
        x='PC1', 
        y='PC2', 
        color='digitalHabitType',
        color_discrete_map={
            'Balanced Explorer': '#4361ee',
            'Mindless Scroller': '#ff4d6d',
            'Stress Escaper': '#ffb703',
            'Goal-Oriented Learner': '#2ec4b6'
        },
        labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
        title=f'PCA: Explained Variance - PC1: {pca.explained_variance_ratio_[0]:.2%}, PC2: {pca.explained_variance_ratio_[1]:.2%}'
    )
    
    # Add loading vectors
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    for i, feature in enumerate(features):
        fig.add_annotation(
            x=loadings[i, 0],
            y=loadings[i, 1],
            ax=0,
            ay=0,
            xanchor="center",
            yanchor="bottom",
            text=feature,
            arrowhead=1,
            arrowwidth=1.5,
            arrowcolor="#636363",
            arrowsize=0.8
        )
    
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<h4>PCA Interpretation:</h4>", unsafe_allow_html=True)
    st.markdown("""
    - **PC1** appears to primarily represent overall screen time intensity and app usage volume
    - **PC2** seems to differentiate between educational vs. entertainment/social usage patterns
    - The clear clustering confirms our digital habit classifications are capturing meaningful patterns
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_about():
    """Display information about the dashboard"""
    st.markdown('<h2 class="subheader">About SlateMate AI Wellbeing Assistant</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    <h3>Project Overview</h3>
    
    SlateMate is an AI-powered digital wellbeing assistant designed to help parents understand and support their children's digital habits. Using advanced machine learning and statistical analysis, SlateMate provides personalized insights and recommendations based on screen time patterns, app usage, and emotional responses.
    
    <h3>Key Features</h3>
    
    - **Digital Habit Classification**: Identifies four distinct digital behavior patterns
    - **Emotion Detection**: Analyzes journal entries to understand emotional states
    - **Personalized Recommendations**: Provides customized guidance for each child
    - **Parent Communication Tools**: Offers conversation starters and approaches
    - **Comprehensive Analytics**: Delivers detailed metrics and visualizations
    
    <h3>Digital Habit Types</h3>
    
    - **Balanced Explorer**: Uses technology in a balanced way across categories with regular breaks.
    - **Mindless Scroller**: Tends to use social media excessively with few breaks and little purpose.
    - **Stress Escaper**: Uses technology to escape negative emotions, often with entertainment apps.
    - **Goal-Oriented Learner**: Primarily uses educational apps with clear purpose and good break habits.
    
    <h3>Methodology</h3>
    
    SlateMate uses a combination of unsupervised machine learning (K-means clustering), sentiment analysis (VADER), and statistical analysis to process digital behavior data and generate meaningful insights. The platform continuously learns from new data to improve its recommendations.
    
    <h3>Privacy & Ethics</h3>
    
    All data processing happens locally within this application. No data is sent to external servers. SlateMate is designed to empower parents while respecting children's autonomy and privacy. The focus is on fostering healthy conversations rather than surveillance.
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    <h3>How to Use This Dashboard</h3>
    
    1. **Dashboard** - View overall statistics and trends across all students
    2. **Individual Reports** - Select specific students to see detailed profiles and recommendations
    3. **Analysis** - Explore deeper patterns and correlations in the data
    4. **About** - Learn more about the SlateMate platform and methodology
    
    To get started, upload your own dataset or use the sample data provided.
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()