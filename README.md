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

## 🔍 Detection Flow (当前检测流程)
1. **Frame Sampling**：`StreamLoader` 按 `SAMPLE_RATE` 获取最新帧。
2. **Motion Detection**：
   - 使用 MOG2 背景建模得到前景掩码。
   - 二值化 + 形态学开运算去噪。
   - 统计所有轮廓面积（单个轮廓需 ≥ `MIN_CONTOUR_AREA`），并计算**总移动面积**。
   - 当总移动面积 ≥ `MOTION_THRESHOLD` 时触发运动，并选取最大轮廓作为 `motion_box`。
3. **YOLO Detection**：识别指定类别目标，得到 `yolo_boxes`。
4. **ROI 合并**：合并 `motion_box` 与 `yolo_boxes`，计算外接矩形作为最终 ROI，进行裁剪。
5. **CLIP 编码**：对 ROI 做图像向量化。
6. **Zero-shot 分类**：基于文本标签推断场景语义（用于解释/过滤）。
7. **异常检测**：短期相似度 + 长期历史验证，输出异常原因。
8. **更新记忆**：写入短期/长期向量库，用于后续对比。

## 🎬 VideoMAE Motion-Only Flow
1. **Frame Sampling**：`StreamLoader` 以 1 FPS 采样连续帧，构建 8 秒滑动窗口。
2. **Motion Detection**：只做运动检测，不裁剪 ROI，也不使用 YOLO。
3. **VideoMAE 编码**：窗口满 8 帧后，对整段视频做 VideoMAE 编码。
4. **异常检测**：短期/长期相似度对比，输出异常原因。
5. **更新记忆**：将 VideoMAE 向量写入短期/长期存储。

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

**2. Run VideoMAE Motion-Only Dashboard**
This launches the VideoMAE + Motion-only pipeline (no YOLO, no ROI crop).
```bash
streamlit run videomae_app.py
```
Access the dashboard at `http://localhost:8501`.

**3. Run CLI Mode**
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
*   `ANOMALY_METHOD`: `cosine` or `zscore` anomaly scoring method.
*   `ZSCORE_THRESHOLD`: Threshold for z-score based detection (higher = less sensitive).
*   `MOTION_THRESHOLD`: Total moving area threshold (sum of motion contour areas).
*   `MIN_CONTOUR_AREA`: Minimum contour area to be counted as motion.
*   `VIDEOMAE_MODEL_NAME`: VideoMAE model ID used in the motion-only app.
*   `VIDEOMAE_CLIP_SIZE`: Sliding window size (seconds / frames at 1 FPS).
*   `VIDEOMAE_SAMPLE_RATE`: Sampling FPS for the VideoMAE sliding window.
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
