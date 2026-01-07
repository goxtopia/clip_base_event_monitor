import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
import cv2
import numpy as np
import base64
import io
from PIL import Image
from datetime import datetime
from sentinel_system import SentinelSystem
from config import settings

# Page Config
st.set_page_config(
    page_title="CLIP-Sentinel Dashboard",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "# CLIP-Sentinel\nIntelligent Visual Sentinel powered by CLIP."
    }
)

# Custom CSS for "Sci-Fi" look
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #c9d1d9;
    }
    .metric-container {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 10px;
        text-align: center;
    }
    .status-normal {
        color: #2ea043;
        font-weight: bold;
        font-size: 24px;
    }
    .status-anomaly {
        color: #da3633;
        font-weight: bold;
        font-size: 24px;
        animation: blink 1s infinite;
    }
    @keyframes blink {
        0% { opacity: 1.0; }
        50% { opacity: 0.5; }
        100% { opacity: 1.0; }
    }
</style>
""", unsafe_allow_html=True)

# Initialize System
@st.cache_resource
def get_system():
    sys = SentinelSystem()
    sys.start()
    return sys

system = get_system()

# Helper to convert frame to base64
def frame_to_base64(frame):
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    # Resize for thumbnail to save bandwidth/memory
    pil_img.thumbnail((160, 120)) 
    buff = io.BytesIO()
    pil_img.save(buff, format="JPEG")
    img_str = base64.b64encode(buff.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

# Sidebar Config
with st.sidebar:
    st.header("Sentinel Config")
    st.info(f"Model: {settings.MODEL_NAME}")
    st.text(f"Device: {settings.DEVICE}")
    
    threshold = st.slider("Anomaly Threshold", 0.0, 1.0, settings.SIMILARITY_THRESHOLD, 0.05)
    settings.SIMILARITY_THRESHOLD = threshold
    
    st.divider()
    st.subheader("Zero-shot Labels")
    labels_input = st.text_area(
        "Enter labels (comma separated)", 
        value=", ".join(settings.ZERO_SHOT_LABELS),
        help="Objects to detect/ignore"
    )
    if st.button("Update Labels"):
        new_labels = [l.strip() for l in labels_input.split(",") if l.strip()]
        if new_labels:
            system.update_labels(new_labels)
            st.success(f"Updated: {new_labels}")
    
    st.divider()
    if st.button("Stop System"):
        system.stop()
        st.stop()

# Layout
col1, col2 = st.columns([2, 1])

# Containers
with col1:
    st.subheader("Live Feed")
    video_placeholder = st.empty()

with col2:
    st.subheader("System Status")
    status_placeholder = st.empty()
    st.divider()
    st.subheader("Change Rate Monitor")
    chart_placeholder = st.empty()
    st.divider()
    st.subheader("Anomaly History")
    history_placeholder = st.empty()

# Session State for History
if "history_df" not in st.session_state:
    st.session_state.history_df = pd.DataFrame(columns=["Snapshot", "Time", "Sim Short", "Sim Long", "Status"])

if "chart_data" not in st.session_state:
    st.session_state.chart_data = pd.DataFrame(columns=["Time", "Similarity"])

# Main Loop
while True:
    result = system.process_step()
    
    if result:
        # 1. Update Video
        frame_display = result['frame'].copy()
        
        # Draw ROI Box
        if result.get('bbox'):
            x, y, w, h = result['bbox']
            # Draw rectangle
            cv2.rectangle(frame_display, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Draw label
            label_text = f"{result['label']} ({result['confidence']:.2f})"
            cv2.putText(frame_display, label_text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Convert BGR to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB")

        # 2. Update Status
        status_text = "ANOMALY DETECTED" if result['is_anomaly'] else "NORMAL"
        status_class = "status-anomaly" if result['is_anomaly'] else "status-normal"
        status_icon = "🚨" if result['is_anomaly'] else "🟢"
        
        status_html = f"""
        <div class="metric-container">
            <div class="{status_class}">
                {status_icon} {status_text}
            </div>
            <p style="margin-top: 5px; font-size: 14px; color: #8b949e;">{result['reason']}</p>
        </div>
        """
        status_placeholder.markdown(status_html, unsafe_allow_html=True)

        # 3. Update Chart (Sci-Fi / Change Rate)
        # "Change Rate" visualized as Inverse Similarity (1 - Sim)
        # Or just plot Similarity with a Threshold Line
        current_time = datetime.fromtimestamp(result['timestamp'])
        new_row = pd.DataFrame({
            "Time": [current_time], 
            "Similarity": [result['sim_short']],
            "LongTerm": [result['sim_long']],
            "Threshold": [settings.SIMILARITY_THRESHOLD]
        })
        st.session_state.chart_data = pd.concat([st.session_state.chart_data, new_row]).tail(100) # Keep last 100 points

        # Create Plotly Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=st.session_state.chart_data["Time"], 
            y=st.session_state.chart_data["Similarity"],
            mode='lines',
            name='Stability (Short)',
            line=dict(color='#00d4ff', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=st.session_state.chart_data["Time"], 
            y=st.session_state.chart_data["LongTerm"],
            mode='lines',
            name='Historical (Long)',
            line=dict(color='#8b949e', width=1, dash='dot')
        ))
        fig.add_trace(go.Scatter(
            x=st.session_state.chart_data["Time"], 
            y=st.session_state.chart_data["Threshold"],
            mode='lines',
            name='Threshold',
            line=dict(color='#da3633', width=2)
        ))
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=0, b=0),
            height=200,
            xaxis=dict(showgrid=False),
            yaxis=dict(range=[0, 1.1], showgrid=True, gridcolor='#30363d')
        )
        chart_placeholder.plotly_chart(fig, use_container_width=True)

        # 4. Update History
        if result['is_anomaly']:
            snapshot_b64 = frame_to_base64(result['frame'])
            hist_row = pd.DataFrame({
                "Snapshot": [snapshot_b64],
                "Time": [current_time.strftime("%H:%M:%S")],
                "Sim Short": [f"{result['sim_short']:.2f}"],
                "Sim Long": [f"{result['sim_long']:.2f}"],
                "Status": ["Anomaly"]
            })
            st.session_state.history_df = pd.concat([hist_row, st.session_state.history_df]).head(10)
            
        history_placeholder.dataframe(
            st.session_state.history_df, 
            hide_index=True, 
            use_container_width=True,
            column_config={
                "Snapshot": st.column_config.ImageColumn(
                    "Snapshot", help="Anomaly Snapshot"
                )
            }
        )

    else:
        time.sleep(0.05)
