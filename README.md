# CLIP-Sentinel (智能视觉哨兵)

## 📖 Overview
CLIP-Sentinel is a real-time intelligent visual anomaly detection system. It leverages the semantic power of **CLIP (Contrastive Language-Image Pre-training)** models to detect visual anomalies in video streams. 

Unlike traditional pixel-based motion detection, CLIP-Sentinel uses semantic vectors to understand the content of the scene. It employs a **dual-memory system**:
1.  **Short-term Memory**: Detects sudden changes against the recent context (last minute).
2.  **Long-term Memory**: Validates changes against historical patterns (same time yesterday/last week) to reduce false positives (e.g., scheduled lighting changes).

## 🏗 System Architecture

```mermaid
graph LR
    A[RTSP Camera] --> B[Stream Sampler]
    B --> C[CLIP Encoder (ViT-B-16-SigLIP2)]
    C --> D[Vector DB (ChromaDB)]
    C --> E[Anomaly Detector]
    D <--> E
    E --> F[WebUI / Alert System]
```

### Key Components
*   **Stream Loader**: Threaded frame capture with latest-frame-only strategy to ensure real-time performance.
*   **Vector Engine**: `open_clip_torch` wrapper utilizing `ViT-B-16-SigLIP2` for high-performance feature extraction.
*   **Memory Store**: `ChromaDB` for persistent vector storage with metadata (timestamp, day, hour).
*   **Detector**: Dual-stage verification logic (Short-term Cosine Similarity + Long-term History Match).
*   **WebUI**: `Streamlit` dashboard with Sci-Fi aesthetics, real-time charts, and visual anomaly history.

## 🚀 Getting Started

### Prerequisites
*   Python 3.10+
*   CUDA-enabled GPU (recommended) or CPU (supported)

### Installation

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

**1. Run the Web Dashboard (Recommended)**
This launches the Sentinel System with a visual interface.
```bash
streamlit run app.py
```
Access the dashboard at `http://localhost:8501`.

**2. Run CLI Mode**
If you only need the backend process with logging:
```bash
python main.py
```

## ⚙️ Configuration
Edit `config.py` to adjust settings:

*   `RTSP_URL`: URL of the video stream (or path to local video file).
*   `SAMPLE_RATE`: Frames per second to process (default: 1.0).
*   `SIMILARITY_THRESHOLD`: Cosine similarity threshold for anomaly detection (default: 0.85).
*   `HISTORY_WINDOW_SIZE`: Number of frames for short-term moving average.
*   `DB_PATH`: Path for ChromaDB persistence.

## 🛠 Tech Stack
*   **Language**: Python
*   **Model**: OpenCLIP (ViT-B-16-SigLIP2)
*   **Vision**: OpenCV
*   **Database**: ChromaDB
*   **UI**: Streamlit, Plotly
*   **Utils**: Pydantic, Loguru

## 📸 Screenshots
The WebUI provides:
*   Real-time video feed.
*   Status indicators (Normal/Anomaly).
*   Live change-rate chart.
*   History log with snapshots of detected anomalies.
