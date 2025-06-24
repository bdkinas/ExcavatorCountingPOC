# Excavator Video Analysis - Scoop Counter

AI-powered system for counting excavator scoops delivered into truck beds using computer vision and YOLOv11.

## Watch the Demo
[![Watch the video](https://youtu.be/FOclvHkpBvY)](https://youtu.be/FOclvHkpBvY)


## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Web Interface (Recommended)
```bash
streamlit run streamlit_scoop_ui.py
```
Open your browser to `http://localhost:8501` and upload your video.

### 3. Command Line Interface
```bash
python tuned_scoop_counter.py video.mp4 --skip 8 --confidence 0.3
```

## 📁 Files

| File | Description |
|------|-------------|
| `streamlit_scoop_ui.py` | **Web interface** - Interactive UI with real-time processing |
| `tuned_scoop_counter.py` | **Command line tool** - Optimized scoop detection algorithm |
| `requirements.txt` | Python dependencies |
| `video.mp4` | Sample excavator video for testing |
| `yolo11n.pt` | YOLOv11 nano model (downloaded automatically) |
| `excavator_video_tonnage_estimator_end_to_end_plan.md` | Original project plan |

## 🎯 Features

### Web Interface
- 🎥 **Video upload** and processing
- 📊 **Real-time statistics** and visualization
- ⚙️ **Adjustable parameters** (confidence, proximity, frame skip)
- 🎯 **Live detection** with bounding boxes and scoop highlights
- 📥 **CSV export** of results
- 📱 **Browser-based** - no installation hassles

### Detection Algorithm
- 🤖 **YOLOv11** for vehicle detection (trucks vs excavators)
- 🎯 **Proximity zones** around stationary trucks
- ⏱️ **Temporal filtering** to prevent over-counting
- 📈 **Zone-based tracking** (entering → over truck → departing)
- 🔧 **Tunable parameters** for different scenarios

## 🛠️ Usage Examples

### Web Interface
1. Run `streamlit run streamlit_scoop_ui.py`
2. Upload your MP4 video
3. Adjust detection parameters if needed
4. Click "Process Video"
5. View real-time detection and export results

### Command Line
```bash
# Basic usage
python tuned_scoop_counter.py video.mp4

# With custom parameters
python tuned_scoop_counter.py video.mp4 --skip 5 --confidence 0.25

# Generate visualization video
python tuned_scoop_counter.py video.mp4 --output results.mp4
```

## ⚙️ Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `--skip` | Process every N frames (higher = faster) | 5 | 1-20 |
| `--confidence` | Detection confidence threshold | 0.2 | 0.1-0.9 |
| `--output` | Save visualization video | None | file path |

## 📊 Performance

- **Processing Speed**: ~2-5x real-time (depending on frame skip)
- **Accuracy**: Temporal filtering eliminates over-counting issues
- **Detection**: YOLOv11 provides 70% faster inference than YOLOv8

## 🔧 Tuning Tips

**Too many scoops detected?**
- Increase `--confidence` (0.3-0.4)
- Increase `--skip` frames (8-10)
- Edit `min_scoop_separation` in code (60→150 frames)

**Too few scoops detected?**
- Decrease `--confidence` (0.15-0.2)  
- Decrease `--skip` frames (3-5)
- Edit `proximity_threshold` in code (220→300 pixels)

## 🏗️ Algorithm Overview

1. **Vehicle Detection**: YOLOv11 identifies trucks and excavators
2. **Truck Classification**: Stationary trucks become "loading zones"
3. **Proximity Tracking**: Excavators entering/exiting truck zones
4. **Temporal Filtering**: Complete cycles (not frame-by-frame detection)
5. **Scoop Validation**: Duration and cooldown checks prevent over-counting

## 🤝 Contributing

This system was developed to solve the over-counting problem in excavator scoop detection through research-based temporal filtering and zone-based tracking approaches.
