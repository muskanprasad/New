import streamlit as st
import pandas as pd
import numpy as np
import base64
import matplotlib.pyplot as plt
from utils import load_data, load_state_monthly_data
from model import create_sequences, build_and_train
from streamlit.components.v1 import html # Import html component

# Set page configuration with a new title for the browser tab
st.set_page_config(page_title="LSTM-Based COVID-19 Forecasting", layout="wide")

# Set background GIF
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(f"""
    <style>
    .stApp {{
        background: linear-gradient(rgba(0,0,0,0.65), rgba(0,0,0,0.65)),
                    url("data:image/gif;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    h1, h2, h3, h4, p, label, .stMarkdown, .stTextInput > div > div > input {{
        color: white !important;
    }}
    /* Sidebar Main Title Styling */
    .sidebar-title-main {{
        font-size: 40px; /* Larger font for COVID-19 */
        font-weight: bold;
        color: white;
        line-height: 1.0;
        margin-bottom: 0;
    }}
    .sidebar-title-sub {{
        font-size: 24px; /* Smaller font for Cases Prediction */
        color: white;
        line-height: 1.0;
        margin-top: 0.2rem;
        margin-bottom: 1.5rem; /* Increased margin for better spacing */
    }}
    /* Sidebar Navigation Links (now text-like) */
    .sidebar-section-text {{
        font-size: 18px; /* Normal text size for sidebar sections */
        color: white;
        margin-bottom: 0.8rem; /* Spacing between text links */
        text-decoration: none; /* Ensure no underline */
        display: block; /* Make it a block element for full width tapping */
        padding: 0.2rem 0; /* Add some padding for better click area if needed */
        transition: color 0.2s ease-in-out; /* Smooth transition for hover */
    }}
    .sidebar-section-text:hover {{
        color: #ADD8E6; /* Light blue on hover for better UX */
    }}
    /* State-wise Cases Heading in Sidebar - Bigger and bolder */
    .sidebar-state-heading {{
        font-size: 26px; /* Larger than normal sidebar text */
        font-weight: bold;
        color: white;
        margin-top: 2rem; /* Add some space above it */
        margin-bottom: 1rem;
    }}
    /* Input field styling in sidebar */
    .sidebar-box input {{
        margin-top: 0.5rem;
        width: 100%;
        border-radius: 8px; /* Rounded corners for input */
        border: 1px solid #ccc;
        padding: 0.5rem;
        background-color: rgba(255, 255, 255, 0.1); /* Semi-transparent background */
    }}
    /* Main Content Heading Styling (similar to image provided) */
    .stHeading {{
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; /* Professional font */
        color: #E0E0E0 !important; /* Lighter color for main headings */
        font-size: 36px; /* Larger size for main headings */
        font-weight: 600; /* Semi-bold */
        border-bottom: 2px solid #555; /* Subtle underline */
        padding-bottom: 10px;
        margin-bottom: 40px; /* Increased margin for better separation */
        margin-top: 50px; /* Space from previous section/top */
    }}
    /* Subheader styling (for "Bar Chart", "Line Chart") */
    h3 {{
        color: #ADD8E6 !important; /* Light blue for subheaders */
        font-size: 24px; /* Slightly larger subheaders */
        margin-top: 25px;
        margin-bottom: 20px;
        font-weight: 500;
    }}
    /* Text within sections - general paragraph text */
    .stMarkdown p, .stMarkdown ul li {{
        font-size: 17px; /* Slightly larger font for readability */
        line-height: 1.7;
        color: #e0e0e0 !important; /* Lighter text color */
        margin-bottom: 10px; /* Spacing between paragraphs/list items */
    }}
    /* Info and Warning boxes */
    .st.success, .st.info, .st.warning, .st.error {{
        border-radius: 8px;
        padding: 15px;
        margin-top: 20px;
        margin-bottom: 20px;
    }}

    /* Removed custom JavaScript for scrolling to revert to default Streamlit anchor behavior */
    </style>
    """, unsafe_allow_html=True)

set_background("COVID-19-Illustration.gif")

# Sidebar
st.sidebar.markdown('<div class="sidebar-title-main">COVID-19</div><div class="sidebar-title-sub">Cases Prediction</div>', unsafe_allow_html=True)

# Reverted links to use standard anchor hrefs for in-page smooth scrolling
st.sidebar.markdown('<a href="#project-overview" class="sidebar-section-text">About Project</a>', unsafe_allow_html=True)
st.sidebar.markdown('<a href="#monthly-new-cases" class="sidebar-section-text">Monthly New Cases</a>', unsafe_allow_html=True)
st.sidebar.markdown('<a href="#cumulative-cases" class="sidebar-section-text">Cumulative Cases</a>', unsafe_allow_html=True)
st.sidebar.markdown('<a href="#predicted-vs-actual-cases" class="sidebar-section-text">Predicted vs Actual</a>', unsafe_allow_html=True)
st.sidebar.markdown('<a href="#state-wise-cases-main" class="sidebar-section-text">State-wise Cases</a>', unsafe_allow_html=True) # THIS IS THE KEY CHANGE REVERTED

st.sidebar.markdown('<p class="sidebar-state-heading">State-wise Data Input</p>', unsafe_allow_html=True)
state_input = st.sidebar.text_input("Enter State Code (e.g., MH, DL)", "").upper()

# Load national-level data
df = load_data()
df = df[(df["location"] == "India") & (df["date"].dt.year == 2025)].copy()
df.sort_values("date", inplace=True)
df["new_cases"] = df["total_cases"].diff().fillna(0)
df["month"] = df["date"].dt.to_period("M").apply(lambda r: r.start_time)

actual_df = df[df["date"].dt.month <= 6]
monthly_df = actual_df.groupby("month").agg({
    "new_cases": "sum",
    "total_cases": "max"
}).reset_index()

daily_cases = actual_df["new_cases"].values
# Ensure create_sequences and build_and_train are properly defined in model.py
X, y, scaler = create_sequences(daily_cases, time_steps=7)
model = build_and_train(X, y)


future_dates = pd.date_range("2025-07-01", "2025-12-31")
future_preds = []
last_seq = daily_cases[-7:] # Get last 7 days for initial prediction input
for _ in range(len(future_dates)):
    seq_scaled = scaler.transform(np.array(last_seq).reshape(-1, 1)).reshape(1, 7, 1)
    pred_scaled = model.predict(seq_scaled)
    pred = scaler.inverse_transform(pred_scaled)[0][0] # Inverse transform to get actual case count
    future_preds.append(pred)
    last_seq = np.append(last_seq[1:], pred) # Update sequence for next prediction

pred_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted": np.round(future_preds) # Round predictions to nearest whole number
})
pred_df["month"] = pred_df["Date"].dt.to_period("M").apply(lambda r: r.start_time)
monthly_pred = pred_df.groupby("month")["Predicted"].sum().reset_index()

# Helper function to draw charts
def draw_charts(data, x_col, y_col, title, color):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(data[x_col], data[y_col], marker='o', color=color, linewidth=2)
    ax.set_title(title, color='#E0E0E0', fontsize=16)
    ax.set_xlabel("Month", color='#E0E0E0', fontsize=12)
    ax.set_ylabel(y_col, color='#E0E0E0', fontsize=12)
    ax.tick_params(axis='x', rotation=45, colors='#E0E0E0')
    ax.tick_params(axis='y', colors='#E0E0E0')
    ax.set_facecolor((40/255, 44/255, 52/255, 0.7))
    fig.patch.set_facecolor("#222831")
    ax.spines['left'].set_color('#555')
    ax.spines['bottom'].set_color('#555')
    ax.spines['right'].set_color('#555')
    ax.spines['top'].set_color('#555')
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='#444')
    plt.tight_layout()
    return fig

# Main Content Sections
st.markdown('<h1 class="stHeading" style="font-size:48px; text-align: center;">An LSTM-Based Predictive Model for Regional COVID-19 Case Load Forecasting</h1>', unsafe_allow_html=True)

st.markdown('<h2 id="project-overview" class="stHeading">🧾 Project Overview</h2>', unsafe_allow_html=True)
st.markdown("""
This project provides an **AI-driven solution** for analyzing and predicting **COVID-19 case trends** in India for the year 2025. It integrates advanced machine learning techniques with an interactive data visualization dashboard.

**Key Objectives:**
-   Analyze monthly new and cumulative COVID-19 cases (Jan–Jun 2025) at a national level.
-   Forecast future COVID-19 cases (Jul–Dec 2025) using a robust time-series model.
-   Offer state-specific case analytics for a more granular understanding of regional trends.

**About LSTM (Long Short-Term Memory):** 
LSTM is a specialized type of Recurrent Neural Network (RNN) architecture, particularly effective for processing, learning from, and making predictions on sequential data, such as time series. Its ability to capture long-term dependencies makes it highly suitable for forecasting complex patterns like disease spread.

**Technologies & Methodologies Used:**
-   Python
-   Streamlit
-   Pandas
-   NumPy
-   TensorFlow
-   LSTM
-   Matplotlib
""")

st.markdown('<h2 id="monthly-new-cases" class="stHeading">📆 Monthly New Cases</h2>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    st.subheader("Bar Chart: Jan-Jun 2025")
    st.bar_chart(monthly_df.set_index("month")["new_cases"])
with col2:
    st.subheader("Line Chart: Jan-Jun 2025")
    st.pyplot(draw_charts(monthly_df, "month", "new_cases", "Monthly New Cases (Actual)", "orange"))

st.markdown('<h2 id="cumulative-cases" class="stHeading">📈 Cumulative Cases</h2>', unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    st.subheader("Bar Chart: Jan-Jun 2025")
    st.bar_chart(monthly_df.set_index("month")["total_cases"])
with col4:
    st.subheader("Line Chart: Jan-Jun 2025")
    st.pyplot(draw_charts(monthly_df, "month", "total_cases", "Cumulative Cases (Actual)", "blue"))

st.markdown('<h2 id="predicted-vs-actual-cases" class="stHeading">🔮 Predicted vs Actual Cases</h2>', unsafe_allow_html=True)
col5, col6 = st.columns(2)
with col5:
    st.subheader("Bar Chart: Jul-Dec 2025 (Predicted)")
    st.bar_chart(monthly_pred.set_index("month")["Predicted"])
with col6:
    st.subheader("Line Chart: Jul-Dec 2025 (Predicted)")
    st.pyplot(draw_charts(monthly_pred, "month", "Predicted", "Predicted Cases", "green"))

# State-wise Cases Section
st.markdown('<h2 id="state-wise-cases-main" class="stHeading">🗺️ State-wise Cases</h2>', unsafe_allow_html=True)

if state_input:
    try:
        state_df = load_state_monthly_data(state_input)
        if not state_df.empty:
            col7, col8 = st.columns(2)
            with col7:
                st.subheader(f"Bar Chart - {state_input} (Jan-Jun 2025)")
                st.bar_chart(state_df.set_index("Date")["Monthly_Cases"])
            with col8:
                st.subheader(f"Line Chart - {state_input} (Jan-Jun 2025)")
                st.pyplot(draw_charts(state_df, "Date", "Monthly_Cases", f"{state_input} Cases", "red"))
        else:
            st.warning("No data available for this state or invalid state code. Please ensure you use a valid two-letter code (e.g., MH, DL).")
    except Exception as e:
        st.error(f"⚠️ An error occurred while loading state data: {e}. Please check the state code and data file.")
else:
    st.info("ℹ️ Enter a valid two-letter state code (e.g., MH for Maharashtra, DL for Delhi) in the sidebar to view specific state data.")