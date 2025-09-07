# 🚦 Enhanced Traffic Management System

An advanced AI-powered traffic management system that uses YOLO (You Only Look Once) object detection to analyze vehicle traffic and optimize signal timing in real-time.

## ✨ Features

### Core Functionality
- **Real-time Vehicle Detection**: Uses YOLOv8 to detect cars, trucks, motorcycles, buses, and bicycles
- **Intelligent Signal Timing**: Optimizes traffic light timing based on vehicle counts and types
- **Multi-Lane Analysis**: Simultaneous processing of up to 4 traffic lanes
- **Web-based Interface**: Intuitive dashboard with animated traffic intersection

### Enhanced Features
- **Weighted Vehicle Priority**: Different vehicle types receive appropriate timing weights
- **Real-time Statistics**: Live monitoring of vehicle counts, processing times, and signal states
- **Progress Tracking**: Visual progress bars and iteration counters
- **Image Processing**: Saves both original and annotated frames for analysis
- **Comprehensive Logging**: Detailed system logs for debugging and monitoring
- **Health Monitoring**: System health checks and status reporting
- **Configurable Parameters**: Adjustable detection thresholds, timing constraints, and analysis intervals

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- 4GB+ RAM recommended
- GPU support optional (for faster processing)

### Setup Instructions

1. **Clone or download the project**
   ```bash
   cd "d:\PYTHON_ML_ALGORITHMS\Traffic_Management\Yolo_Web_Video - 2 - Copy"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download YOLO models** (if not present)
   - The application will automatically download `yolov8s.pt` on first run
   - For better accuracy, you can use `yolov8x.pt` (larger model)
   - For faster processing, you can use `yolov8n.pt` (smaller model)

4. **Configure the system** (optional)
   - Edit `config.json` to adjust detection parameters
   - Modify confidence thresholds, timing constraints, etc.

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the web interface**
   - Open your browser and go to `http://localhost:5000`
   - The system will perform a health check on startup

## 📱 Usage Guide

### Getting Started
1. **Upload Videos**: Click on the camera icons in each quadrant to upload video files
   - Supported formats: MP4, AVI, MOV, MKV, WMV
   - Maximum file size: 100MB per video
   - Upload at least one video to start analysis

2. **Configure Analysis** (optional):
   - **Analysis Interval**: Time between frames (1-60 seconds)
   - **Max Iterations**: Maximum number of analysis cycles (1-100)
   - **Confidence Threshold**: Detection confidence level (0.1-1.0)

3. **Start Analysis**: Click the "🚀 Start Analysis" button
   - The system will upload videos, start processing, and display real-time results
   - Progress bar shows completion status
   - Live statistics update during analysis

4. **Monitor Results**:
   - View real-time vehicle counts for each lane
   - See optimized signal timing recommendations
   - Track processing performance and statistics

### Key Metrics
- **Vehicle Counts**: Real-time detection of different vehicle types
- **Signal Timing**: Optimized green light durations for each lane
- **Priority Order**: Lanes ranked by traffic volume
- **Processing Time**: Performance metrics for each analysis iteration

## 🔧 Configuration

### System Configuration (`config.json`)
```json
{
  "yolo": {
    "model": "yolov8s.pt",
    "confidence_threshold": 0.4,
    "iou_threshold": 0.5
  },
  "analysis": {
    "interval_seconds": 5,
    "max_iterations": 20,
    "min_green_time": 5.0,
    "max_green_time": 60.0
  }
}
```

### Vehicle Weights
The system uses weighted scoring for different vehicle types:
- **Cars**: 1.0x (baseline)
- **Trucks**: 2.0x (need more time)
- **Buses**: 2.5x (highest priority)
- **Motorcycles**: 0.5x (faster movement)
- **Bicycles**: 0.3x (minimal impact)

## 📊 API Endpoints

### Main Routes
- `GET /` - Main dashboard interface
- `POST /upload_videos` - Upload video files
- `POST /start_analysis` - Begin traffic analysis
- `POST /stop_analysis` - Stop ongoing analysis
- `GET /latest_data` - Get current analysis results

### System Routes
- `GET /health` - System health check
- `GET /analysis_status` - Current analysis status
- `GET /results_history` - Historical analysis data
- `GET /config` - System configuration
- `POST /config` - Update configuration
- `POST /reset` - Reset system state

## 🏗️ Architecture

### Backend Components
- **Flask Web Server**: Handles HTTP requests and serves the interface
- **YOLO Detection Engine**: Processes video frames for vehicle detection
- **Signal Optimization**: Calculates optimal timing based on traffic data
- **State Management**: Thread-safe handling of system state
- **Logging System**: Comprehensive logging for debugging and monitoring

### Frontend Components
- **Interactive Dashboard**: Real-time traffic intersection visualization
- **File Upload Interface**: Drag-and-drop video upload with validation
- **Progress Monitoring**: Live updates of analysis progress
- **Results Display**: Dynamic rendering of detection results and timing data

### Data Flow
1. Videos uploaded through web interface
2. Backend processes frames using YOLO detection
3. Vehicle counts feed into signal optimization algorithm
4. Results stored and served to frontend
5. Real-time updates displayed in dashboard

## 🚀 Performance Optimization

### Speed Improvements
- **Model Selection**: Use `yolov8n.pt` for faster processing
- **Frame Sampling**: Adjust analysis interval for real-time performance
- **Threading**: Background processing doesn't block the UI
- **Caching**: Results cached to prevent duplicate processing

### Accuracy Improvements
- **Model Selection**: Use `yolov8x.pt` for better detection accuracy
- **Confidence Tuning**: Adjust thresholds based on video quality
- **Multiple Classes**: Detects various vehicle types for better planning

## 🐛 Troubleshooting

### Common Issues

**Videos not uploading**
- Check file format (must be MP4, AVI, MOV, MKV, or WMV)
- Verify file size is under 100MB
- Ensure stable internet connection

**Detection not working**
- Verify YOLO model downloaded correctly
- Check video quality and lighting
- Adjust confidence threshold in settings

**Analysis stops unexpectedly**
- Check system logs in `traffic_system.log`
- Verify sufficient memory available
- Restart the application if needed

**Poor signal timing results**
- Upload higher quality videos
- Increase confidence threshold
- Ensure videos show actual traffic scenes

### Performance Issues
- **High CPU usage**: Use smaller YOLO model (yolov8n.pt)
- **Memory issues**: Reduce max iterations or video resolution
- **Slow processing**: Increase analysis interval between frames

## 🔒 Security Considerations

- File upload validation prevents malicious files
- Secure filename handling prevents directory traversal
- Input validation on all API endpoints
- No external network access required for core functionality

## 📁 Project Structure

```
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── config.json           # System configuration
├── templates/
│   └── index.html        # Main web interface
├── uploads/              # Uploaded video files
├── results/              # Analysis results and images
├── static_results/       # Web-served result images
├── yolov8s.pt           # YOLO model file
└── traffic_system.log   # Application logs
```

## 🤝 Contributing

To improve this system:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is provided as-is for educational and research purposes.

## 🙋‍♂️ Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the application logs in `traffic_system.log`
3. Verify your system meets the minimum requirements
4. Test with the provided sample videos first

---

## 🎯 Quick Start Example

1. **Run the application**:
   ```bash
   python app.py
   ```

2. **Open browser**: Go to `http://localhost:5000`

3. **Upload a video**: Click any camera icon and select a traffic video

4. **Start analysis**: Click "🚀 Start Analysis"

5. **View results**: Watch real-time detection and signal optimization

The system will process your video, detect vehicles, and provide optimized traffic signal timing recommendations with a visual dashboard showing the results.
