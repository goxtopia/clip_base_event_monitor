import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import cv2
from datetime import datetime

from config import settings
from videomae_system import VideoMAESentinelSystem


st.set_page_config(
    page_title="VideoMAE Sentinel",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
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
            font-size: 22px;
        }
        .status-anomaly {
            color: #da3633;
            font-weight: bold;
            font-size: 22px;
            animation: blink 1s infinite;
        }
        @keyframes blink {
            0% { opacity: 1.0; }
            50% { opacity: 0.5; }
            100% { opacity: 1.0; }
        }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def get_system():
    system = VideoMAESentinelSystem()
    system.start()
    return system


system = get_system()

with st.sidebar:
    st.header("VideoMAE Config")
    st.info(f"Model: {settings.VIDEOMAE_MODEL_NAME}")
    st.text(f"Device: {settings.DEVICE}")

    threshold = st.slider("Anomaly Threshold", 0.0, 1.0, settings.SIMILARITY_THRESHOLD, 0.05)
    settings.SIMILARITY_THRESHOLD = threshold

    anomaly_method = st.selectbox(
        "Anomaly Method",
        ["cosine", "zscore"],
        index=["cosine", "zscore"].index(settings.ANOMALY_METHOD)
    )
    settings.ANOMALY_METHOD = anomaly_method
    if settings.ANOMALY_METHOD == "zscore":
        zscore_threshold = st.slider("Z-Score Threshold", 0.5, 5.0, settings.ZSCORE_THRESHOLD, 0.1)
        settings.ZSCORE_THRESHOLD = zscore_threshold

    st.divider()
    st.subheader("Motion Detection")
    motion_blur = st.slider("Denoise Blur Size", 1, 51, settings.MOTION_BLUR_SIZE, 2)
    motion_thresh = st.slider("Motion Area Threshold (px²)", 0.0, 50000.0, settings.MOTION_THRESHOLD, 100.0)
    system.motion_detector.update_settings(motion_blur, motion_thresh)

    st.divider()
    st.subheader("Video Clip")
    st.text(f"Window: {settings.VIDEOMAE_CLIP_SIZE}s @ {settings.VIDEOMAE_SAMPLE_RATE} FPS")

    st.divider()
    if st.button("Stop System"):
        system.stop()
        st.stop()


col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Live Feed")
    video_placeholder = st.empty()

with col2:
    st.subheader("System Status")
    status_placeholder = st.empty()
    st.divider()
    st.subheader("Similarity Monitor")
    chart_placeholder = st.empty()

if "chart_data" not in st.session_state:
    st.session_state.chart_data = pd.DataFrame(columns=["Time", "Similarity", "LongTerm", "Threshold"])

while True:
    result = system.process_step()

    if result:
        frame_display = result["frame"].copy()
        frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB")

        status_text = "ANOMALY DETECTED" if result["is_anomaly"] else "NORMAL"
        status_class = "status-anomaly" if result["is_anomaly"] else "status-normal"
        status_icon = "🚨" if result["is_anomaly"] else "🟢"
        detail = result["status"]

        status_html = f"""
        <div class="metric-container">
            <div class="{status_class}">
                {status_icon} {status_text}
            </div>
            <p style="margin-top: 5px; font-size: 14px; color: #8b949e;">{detail}</p>
        </div>
        """
        status_placeholder.markdown(status_html, unsafe_allow_html=True)

        if result["sim_short"] is not None:
            current_time = datetime.fromtimestamp(result["timestamp"])
            new_row = pd.DataFrame({
                "Time": [current_time],
                "Similarity": [result["sim_short"]],
                "LongTerm": [result["sim_long"]],
                "Threshold": [settings.SIMILARITY_THRESHOLD]
            })
            st.session_state.chart_data = pd.concat(
                [st.session_state.chart_data, new_row]
            ).tail(100)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=st.session_state.chart_data["Time"],
            y=st.session_state.chart_data["Similarity"],
            mode="lines",
            name="Short-term Similarity"
        ))
        fig.add_trace(go.Scatter(
            x=st.session_state.chart_data["Time"],
            y=st.session_state.chart_data["LongTerm"],
            mode="lines",
            name="Long-term Similarity"
        ))
        fig.add_trace(go.Scatter(
            x=st.session_state.chart_data["Time"],
            y=st.session_state.chart_data["Threshold"],
            mode="lines",
            name="Threshold"
        ))
        fig.update_layout(
            template="plotly_dark",
            height=260,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        chart_placeholder.plotly_chart(fig, use_container_width=True)
