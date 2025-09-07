import os
import time
import threading
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from ultralytics import YOLO
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('traffic_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class VehicleCounts:
    """Data class for vehicle counts"""
    car: int = 0
    truck: int = 0
    motorcycle: int = 0
    bus: int = 0
    bicycle: int = 0
    # Emergency vehicles
    ambulance: int = 0
    firetruck: int = 0
    
    def total(self) -> int:
        return (
            self.car
            + self.truck
            + self.motorcycle
            + self.bus
            + self.bicycle
            + self.ambulance
            + self.firetruck
        )
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class SignalPlan:
    """Data class for traffic signal plan"""
    cycle: float
    greens: List[float]
    priority: List[int]
    total_vehicles: int
    timestamp: str

@dataclass
class Config:
    """Configuration class"""
    UPLOAD_FOLDER: str = "uploads"
    RESULTS_FOLDER: str = "results"
    STATIC_RESULTS_FOLDER: str = "static_results"
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS: set = None
    YOLO_MODEL: str = "yolov8s.pt"
    TARGET_CLASSES: List[str] = None
    CONFIDENCE_THRESHOLD: float = 0.4
    IOU_THRESHOLD: float = 0.5
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 480
    ANALYSIS_INTERVAL: int = 5
    MAX_ITERATIONS: int = 20
    MIN_GREEN_TIME: float = 5.0
    MAX_GREEN_TIME: float = 60.0
    DEFAULT_CYCLE_TIME: float = 60.0
    # Emergency detection options
    USE_EMERGENCY_HEURISTIC: bool = True
    EMERGENCY_MODEL: Optional[str] = None
    EMERGENCY_LIGHT_PIXEL_RATIO: float = 0.005  # fraction of roof pixels to count as light
    
    def __post_init__(self):
        if self.ALLOWED_EXTENSIONS is None:
            self.ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv'}
        if self.TARGET_CLASSES is None:
            # Include emergency vehicle classes
            self.TARGET_CLASSES = [
                "car",
                "truck",
                "motorcycle",
                "bus",
                "bicycle",
                "ambulance",
                "firetruck",
            ]

# Initialize configuration
config = Config()

# Base directory (absolute) to avoid issues when running from different CWDs
BASE_DIR = Path(__file__).resolve().parent

# Convert relative folders in config to absolute paths
config.UPLOAD_FOLDER = str((BASE_DIR / config.UPLOAD_FOLDER).resolve())
config.RESULTS_FOLDER = str((BASE_DIR / config.RESULTS_FOLDER).resolve())
config.STATIC_RESULTS_FOLDER = str(
    (BASE_DIR / config.STATIC_RESULTS_FOLDER).resolve()
)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = config.MAX_FILE_SIZE

# Create directories
for folder in [
    config.UPLOAD_FOLDER,
    config.RESULTS_FOLDER,
    config.STATIC_RESULTS_FOLDER,
]:
    Path(folder).mkdir(parents=True, exist_ok=True)
    logger.info(f"Created/verified directory: {folder}")

# Initialize YOLO model with error handling
model = None
try:
    # Allow skipping YOLO load (useful for unit tests or CI) by setting SKIP_YOLO=1
    if os.environ.get("SKIP_YOLO", "0") != "1":
        model = YOLO(config.YOLO_MODEL)
        logger.info(f"Successfully loaded YOLO model: {config.YOLO_MODEL}")
    else:
        logger.info("Skipping YOLO model load due to SKIP_YOLO env var")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {e}")
    raise

# Global state management
class TrafficSystemState:
    """Centralized state management for the traffic system"""
    
    def __init__(self):
        self.video_paths: List[str] = []
        self.latest_result: Dict = {}
        self.stop_thread: bool = False
        self.analysis_running: bool = False
        self.analysis_thread: Optional[threading.Thread] = None
        self.results_history: List[Dict] = []
        self.lock = threading.Lock()
    
    def reset(self):
        """Reset the system state"""
        with self.lock:
            self.video_paths = []
            self.latest_result = {}
            self.stop_thread = False
            self.analysis_running = False
            self.results_history = []
    
    def add_result(self, result: Dict):
        """Add a result to history with size limit"""
        with self.lock:
            self.results_history.append(result)
            # Keep only last 50 results to prevent memory issues
            if len(self.results_history) > 50:
                self.results_history = self.results_history[-50:]

# Initialize global state
system_state = TrafficSystemState()

# Utility functions
def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS

def validate_video_file(file) -> Tuple[bool, str]:
    """Validate uploaded video file"""
    if not file or file.filename == '':
        return False, "No file selected"
    
    if not allowed_file(file.filename):
        allowed = ', '.join(config.ALLOWED_EXTENSIONS)
        return False, (
            f"File type not allowed. Allowed types: {allowed}"
        )
    
    return True, "Valid file"

def safe_video_capture(video_path: str) -> Optional[cv2.VideoCapture]:
    """Safely create video capture with error handling"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return None
        return cap
    except Exception as e:
        logger.error(f"Error creating video capture for {video_path}: {e}")
        return None


# -------------------------
# Traffic signal planning
# -------------------------
def compute_signal_plan(
    counts_list: List[VehicleCounts],
    cycle: float = None,
    min_green: float = None,
    max_green: float = None,
) -> SignalPlan:
    """
    Compute optimized traffic signal timing plan based on vehicle counts
    
    Args:
        counts_list: List of VehicleCounts for each lane
        cycle: Total cycle time (uses config default if None)
        min_green: Minimum green time per lane (uses config default if None)
        max_green: Maximum green time per lane (uses config default if None)
    
    Returns:
        SignalPlan object with timing information
    """
    if cycle is None:
        cycle = config.DEFAULT_CYCLE_TIME
    if min_green is None:
        min_green = config.MIN_GREEN_TIME
    if max_green is None:
        max_green = config.MAX_GREEN_TIME
    
    # Calculate total vehicles across all lanes
    total_vehicles = sum(counts.total() for counts in counts_list) or 1
    
    # Calculate proportional green times
    greens = []
    for counts in counts_list:
        # Base calculation on total vehicles in lane
        lane_vehicles = counts.total()

        # Apply weighting for different vehicle types
        # Emergency vehicles have much higher weight to prioritize them
        weighted_count = (
            counts.car * 1.0
            + counts.truck * 2.0
            + counts.bus * 2.5
            + counts.motorcycle * 0.5
            + counts.bicycle * 0.3
            + counts.ambulance * 5.0
            + counts.firetruck * 5.0
        )
        
        # Calculate proportional time
        if total_vehicles > 0:
            proportion = weighted_count / total_vehicles
            green_time = proportion * cycle
        else:
            green_time = cycle / len(counts_list)  # Equal distribution
        
        # Apply constraints
        green_time = max(min_green, min(green_time, max_green))
        greens.append(round(green_time, 1))
    
    # Create priority order (highest vehicle count first)
    priority = sorted(range(len(counts_list)), 
                     key=lambda i: counts_list[i].total(), 
                     reverse=True)
    
    return SignalPlan(
        cycle=cycle,
        greens=greens,
        priority=priority,
        total_vehicles=int(total_vehicles),
        timestamp=datetime.now().isoformat()
    )


def detect_vehicles_in_frame(frame: np.ndarray) -> Tuple[np.ndarray, VehicleCounts]:
    """
    Detect vehicles in a single frame using YOLO
    
    Args:
        frame: Input video frame
        
    Returns:
        Tuple of (processed_frame, vehicle_counts)
    """
    counts = VehicleCounts()
    processed_frame = frame.copy()
    
    try:
        # Run YOLO detection
        results = model.predict(
            frame, 
            imgsz=config.FRAME_WIDTH,
            conf=config.CONFIDENCE_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            verbose=False
        )
        
        if not results or len(results) == 0:
            logger.debug("No YOLO results returned")
            return processed_frame, counts
            
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            logger.debug("No boxes detected in frame")
            return processed_frame, counts
        # Process detections
        detection_count = 0

        # Helper: roof-light heuristic to detect emergency vehicles when
        # a class is not present in model predictions
        def _detect_roof_lights(crop: np.ndarray) -> Tuple[bool, float]:
            """Return (is_emergency_light, ratio_of_light_pixels).
            Looks for red/blue bright pixels in the roof region.
            """
            try:
                if crop is None or crop.size == 0:
                    return False, 0.0
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                # red mask (two ranges) and blue mask
                lower_red1 = np.array([0, 100, 150])
                upper_red1 = np.array([10, 255, 255])
                lower_red2 = np.array([160, 100, 150])
                upper_red2 = np.array([179, 255, 255])
                lower_blue = np.array([90, 80, 150])
                upper_blue = np.array([140, 255, 255])

                mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
                mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
                mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

                mask = cv2.bitwise_or(mask_red1, mask_red2)
                mask = cv2.bitwise_or(mask, mask_blue)

                light_pixels = int(np.count_nonzero(mask))
                total_pixels = crop.shape[0] * crop.shape[1]
                ratio = (
                    light_pixels / float(total_pixels)
                    if total_pixels > 0
                    else 0.0
                )
                is_emergency = ratio >= config.EMERGENCY_LIGHT_PIXEL_RATIO
                return bool(is_emergency), float(ratio)
            except Exception:
                return False, 0.0

        for box in result.boxes:
            try:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                confidence = float(box.conf[0])
                
                # Count vehicles of target classes
                if label in config.TARGET_CLASSES:
                    detection_count += 1
                    
                    # Update counts based on label
                    if label == 'car':
                        counts.car += 1
                    elif label == 'truck':
                        counts.truck += 1
                    elif label == 'motorcycle':
                        counts.motorcycle += 1
                    elif label == 'bus':
                        counts.bus += 1
                    elif label == 'bicycle':
                        counts.bicycle += 1
                    elif label == 'ambulance':
                        counts.ambulance += 1
                    elif label == 'firetruck':
                        counts.firetruck += 1
                    
                    # Draw bounding box and label
                    # box.xyxy may be nested; safely extract coordinates
                    try:
                        coords = box.xyxy[0]
                        x1, y1, x2, y2 = map(int, coords.tolist())
                    except Exception:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Color coding by vehicle type
                    colors = {
                        'car': (0, 255, 0),      # Green
                        'truck': (0, 0, 255),    # Red
                        'bus': (255, 0, 0),      # Blue
                        'motorcycle': (255, 255, 0),  # Cyan
                        'bicycle': (0, 255, 255),     # Yellow
                        'ambulance': (0, 128, 255),   # Orange-ish
                        'firetruck': (0, 0, 128),     # Dark red
                    }
                    color = colors.get(label, (0, 255, 0))
                    
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label with confidence
                    label_text = f"{label}: {confidence:.2f}"
                    cv2.putText(processed_frame, label_text, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    # Heuristic: if model did not classify as ambulance/firetruck but
                    # config allows heuristic detection, crop roof area and try to detect
                    # emergency lights (red/blue). This helps when using generic vehicle
                    # classes only.
                    try:
                        if config.USE_EMERGENCY_HEURISTIC and label in ('car', 'truck', 'bus'):
                            # Compute roof crop (top ~20% of bbox)
                            h = max(1, y2 - y1)
                            roof_h = max(6, int(h * 0.18))
                            ry1 = max(0, y1 - roof_h)  # slightly above bbox
                            ry2 = min(processed_frame.shape[0], y1 + int(h * 0.1))
                            rx1 = max(0, x1)
                            rx2 = min(processed_frame.shape[1], x2)
                            roof_crop = frame[ry1:ry2, rx1:rx2]
                            found_light, light_ratio = _detect_roof_lights(roof_crop)
                            if found_light:
                                # Upgrade label to ambulance/firetruck depending on context
                                # If the vehicle color or shape suggests truck, mark firetruck
                                # else mark ambulance. This is a heuristic fallback.
                                alt_label = 'ambulance'
                                if label == 'truck' or (x2 - x1) > 150:
                                    alt_label = 'firetruck'
                                # increment corresponding count and annotate
                                if alt_label == 'ambulance':
                                    counts.ambulance += 1
                                else:
                                    counts.firetruck += 1
                                # annotate change
                                cv2.putText(processed_frame, f"{alt_label} (light){light_ratio:.3f}", (x1, y2 + 20),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 255), 2)
                                logger.debug(f"Heuristic detected {alt_label} at ratio={light_ratio:.4f}")
                    except Exception as e:
                        logger.debug(f"Roof-light heuristic failed: {e}")
                               
            except Exception as e:
                logger.error(f"Error processing detection box: {e}")
                continue
                
        logger.debug(f"Detected {detection_count} vehicles in frame")
                
    except Exception as e:
        logger.error(f"Error in vehicle detection: {e}")
    
    return processed_frame, counts

# -------------------------
# Enhanced background analysis
# -------------------------
def analyze_videos(interval: int = None, max_iterations: int = None) -> None:
    """
    Enhanced video analysis with better error handling and performance
    
    Args:
        interval: Time between analysis iterations in seconds
        max_iterations: Maximum number of analysis iterations
    """
    if interval is None:
        interval = config.ANALYSIS_INTERVAL
    if max_iterations is None:
        max_iterations = config.MAX_ITERATIONS
        
    system_state.analysis_running = True
    logger.info(f"Starting video analysis: {len(system_state.video_paths)} videos, "
               f"{interval}s interval, {max_iterations} max iterations")
    
    # Initialize video captures
    video_captures = []
    for video_path in system_state.video_paths:
        cap = safe_video_capture(video_path)
        if cap is not None:
            video_captures.append(cap)
        else:
            logger.warning(f"Failed to open video: {video_path}")
            # Create a dummy capture for consistency
            video_captures.append(None)
    
    if not any(video_captures):
        logger.error("No valid video captures available")
        system_state.analysis_running = False
        return
    
    iteration = 0
    
    try:
        while iteration < max_iterations and not system_state.stop_thread:
            start_time = time.time()
            logger.info(f"Starting analysis iteration {iteration + 1}/{max_iterations}")
            
            frames = []
            originals = []
            all_counts = []
            
            # Capture frames from all videos
            for i, cap in enumerate(video_captures):
                if cap is None:
                    # Create black frame for missing video
                    frame = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), 
                                   dtype=np.uint8)
                else:
                    try:
                        # Set frame position based on iteration
                        frame_time_ms = iteration * interval * 1000
                        cap.set(cv2.CAP_PROP_POS_MSEC, frame_time_ms)
                        ret, frame = cap.read()
                        
                        if not ret:
                            logger.warning(f"Could not read frame from video {i}")
                            frame = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), 
                                           dtype=np.uint8)
                        else:
                            frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
                            
                    except Exception as e:
                        logger.error(f"Error reading frame from video {i}: {e}")
                        frame = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), 
                                       dtype=np.uint8)
                
                originals.append(frame.copy())
                frames.append(frame)
            
            # Process frames for vehicle detection
            processed_frames = []
            for i, frame in enumerate(frames):
                try:
                    processed_frame, counts = detect_vehicles_in_frame(frame)
                    processed_frames.append(processed_frame)
                    all_counts.append(counts)
                    logger.debug(f"Lane {i+1}: {counts.total()} vehicles detected")
                except Exception as e:
                    logger.error(f"Error processing frame {i}: {e}")
                    processed_frames.append(frame)
                    all_counts.append(VehicleCounts())
            
            # Compute signal timing plan
            try:
                signal_plan = compute_signal_plan(all_counts)
                logger.info(f"Signal plan computed: {signal_plan.greens}s green times")
            except Exception as e:
                logger.error(f"Error computing signal plan: {e}")
                signal_plan = SignalPlan(
                    cycle=config.DEFAULT_CYCLE_TIME,
                    greens=[config.MIN_GREEN_TIME] * len(all_counts),
                    priority=list(range(len(all_counts))),
                    total_vehicles=0,
                    timestamp=datetime.now().isoformat()
                )
            
            # Save processed images with enhanced annotations
            try:
                for i, (processed_frame, original_frame, counts) in enumerate(
                    zip(processed_frames, originals, all_counts)):
                    
                    # Add comprehensive annotations
                    _add_frame_annotations(
                        processed_frame, i, counts, signal_plan, iteration + 1
                    )
                    
                    # Save processed frame
                    lane_path = Path(config.RESULTS_FOLDER) / f"lane{i+1}.jpg"
                    cv2.imwrite(str(lane_path), processed_frame)
                    
                    # Save original frame
                    orig_path = Path(config.RESULTS_FOLDER) / f"lane{i+1}_orig.jpg"
                    cv2.imwrite(str(orig_path), original_frame)
                    
                    # Also save to static results for web serving
                    static_lane_path = Path(config.STATIC_RESULTS_FOLDER) / f"lane{i+1}.jpg"
                    static_orig_path = Path(config.STATIC_RESULTS_FOLDER) / f"lane{i+1}_orig.jpg"
                    cv2.imwrite(str(static_lane_path), processed_frame)
                    cv2.imwrite(str(static_orig_path), original_frame)
                    
                logger.debug("All frames saved successfully")
                    
            except Exception as e:
                logger.error(f"Error saving frames: {e}")
            
            # Update system state with results
            result_data = {
                "iteration": iteration + 1,
                "counts": [counts.to_dict() for counts in all_counts],
                "cycle": signal_plan.cycle,
                "greens": signal_plan.greens,
                "priority": signal_plan.priority,
                "total": signal_plan.total_vehicles,
                "timestamp": signal_plan.timestamp,
                "processing_time": round(time.time() - start_time, 2)
            }
            
            system_state.latest_result = result_data
            system_state.add_result(result_data)
            
            logger.info(f"Iteration {iteration + 1} completed in {result_data['processing_time']}s. "
                       f"Total vehicles: {signal_plan.total_vehicles}")
            
            iteration += 1
            
            # Sleep for the remaining interval time
            elapsed = time.time() - start_time
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    except Exception as e:
        logger.error(f"Critical error in analysis loop: {e}")
    finally:
        # Clean up video captures
        for cap in video_captures:
            if cap is not None:
                try:
                    cap.release()
                except Exception as e:
                    logger.error(f"Error releasing video capture: {e}")
        
        system_state.analysis_running = False
        logger.info("Video analysis completed")


def _add_frame_annotations(frame: np.ndarray, lane_id: int, counts: VehicleCounts, 
                          signal_plan: SignalPlan, iteration: int) -> None:
    """Add comprehensive annotations to processed frame"""
    try:
        height, width = frame.shape[:2]
        
        # Background for text
        overlay = frame.copy()
        
        # Lane header
        cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Lane title
        cv2.putText(frame, f"LANE {lane_id + 1}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Vehicle counts
        count_text = f"Cars:{counts.car} Trucks:{counts.truck} Bikes:{counts.motorcycle}"
        if counts.bus > 0:
            count_text += f" Buses:{counts.bus}"
        if counts.bicycle > 0:
            count_text += f" Bicycles:{counts.bicycle}"
        if counts.ambulance > 0:
            count_text += f" Ambulance:{counts.ambulance}"
        if counts.firetruck > 0:
            count_text += f" Firetruck:{counts.firetruck}"
            
        cv2.putText(frame, count_text, (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Signal timing
        green_time = signal_plan.greens[lane_id] if lane_id < len(signal_plan.greens) else 0
        timing_text = f"Green: {green_time}s | Total: {counts.total()}"
        cv2.putText(frame, timing_text, (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Iteration number (bottom right)
        cv2.putText(frame, f"#{iteration}", (width - 80, height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    except Exception as e:
        logger.error(f"Error adding frame annotations: {e}")


def create_test_frame_with_annotations(lane_id: int) -> np.ndarray:
    """Create a test frame with annotations for debugging"""
    # Create a test frame
    frame = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
    frame[:] = (50, 50, 50)  # Dark gray background
    
    # Add some test rectangles to simulate vehicles
    cv2.rectangle(frame, (50, 100), (150, 200), (0, 255, 0), 2)  # Car
    cv2.rectangle(frame, (200, 150), (350, 250), (0, 0, 255), 2)  # Truck
    
    # Create test counts
    counts = VehicleCounts()
    counts.car = 2
    counts.truck = 1
    counts.motorcycle = 0
    counts.bus = 0
    counts.bicycle = 0
    
    # Create test signal plan
    signal_plan = SignalPlan(
        cycle=60.0,
        greens=[15.0, 20.0, 10.0, 15.0],
        priority=[1, 0, 3, 2],
        total_vehicles=3,
        timestamp=datetime.now().isoformat()
    )
    
    # Add annotations
    _add_frame_annotations(frame, lane_id, counts, signal_plan, 1)
    
    return frame

# -------------------------
# Enhanced Flask Routes
# -------------------------
@app.route("/test_images")
def test_images():
    """Generate test images to verify image processing and serving"""
    try:
        logger.info("Generating test images...")
        
        # Create test images for each lane
        for i in range(4):
            # Create test frame
            test_frame = create_test_frame_with_annotations(i)
            
            # Save to both directories
            lane_path = Path(config.RESULTS_FOLDER) / f"lane{i+1}.jpg"
            static_lane_path = Path(config.STATIC_RESULTS_FOLDER) / f"lane{i+1}.jpg"
            
            cv2.imwrite(str(lane_path), test_frame)
            cv2.imwrite(str(static_lane_path), test_frame)
            
            # Create a simple original frame (just colored background)
            orig_frame = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
            orig_frame[:] = (100, 150, 200)  # Light blue background
            cv2.putText(orig_frame, f"ORIGINAL LANE {i+1}", (50, config.FRAME_HEIGHT//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            orig_path = Path(config.RESULTS_FOLDER) / f"lane{i+1}_orig.jpg"
            static_orig_path = Path(config.STATIC_RESULTS_FOLDER) / f"lane{i+1}_orig.jpg"
            
            cv2.imwrite(str(orig_path), orig_frame)
            cv2.imwrite(str(static_orig_path), orig_frame)
        
        logger.info("Test images generated successfully")
        return jsonify({
            "success": True,
            "message": "Test images generated for all 4 lanes",
            "files_created": 8
        })
        
    except Exception as e:
        logger.error(f"Error generating test images: {e}")
        return jsonify({
            "success": False,
            "message": f"Failed to generate test images: {str(e)}"
        }), 500


@app.route("/")
def index():
    """Serve the main traffic management interface"""
    try:
        return render_template("index.html")
    except Exception as e:
        logger.error(f"Error serving index page: {e}")
        return jsonify({"error": "Failed to load page"}), 500


@app.route("/upload_videos", methods=["POST"])
def upload_videos():
    """
    Handle video uploads with enhanced validation and error handling
    """
    try:
        system_state.video_paths = []
        uploaded_count = 0
        errors = []
        
        # Video mapping for the 4 quadrants
        video_mapping = {
            'video-tl': 'lane1.mp4',    # top-left
            'video-tr': 'lane2.mp4',    # top-right  
            'video-bl': 'lane3.mp4',    # bottom-left
            'video-br': 'lane4.mp4'     # bottom-right
        }
        
        for video_id, filename in video_mapping.items():
            if video_id in request.files:
                file = request.files[video_id]
                
                # Validate file
                is_valid, message = validate_video_file(file)
                if not is_valid:
                    errors.append(f"{video_id}: {message}")
                    continue
                
                try:
                    # Secure the filename
                    secure_name = secure_filename(filename)
                    file_path = Path(config.UPLOAD_FOLDER) / secure_name
                    
                    # Save file
                    file.save(str(file_path))
                    
                    # Verify the saved file can be opened
                    test_cap = safe_video_capture(str(file_path))
                    if test_cap is None:
                        errors.append(f"{video_id}: Uploaded file is not a valid video")
                        file_path.unlink(missing_ok=True)  # Delete invalid file
                        continue
                    test_cap.release()
                    
                    system_state.video_paths.append(str(file_path))
                    uploaded_count += 1
                    logger.info(f"Successfully uploaded: {video_id} -> {secure_name}")
                    
                except Exception as e:
                    error_msg = f"{video_id}: Upload failed - {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
        
        if uploaded_count == 0:
            return jsonify({
                "success": False, 
                "message": "No videos were successfully uploaded",
                "errors": errors
            }), 400
        
        response_data = {
            "success": True,
            "message": f"Successfully uploaded {uploaded_count} video(s)",
            "uploaded_count": uploaded_count,
            "video_paths": system_state.video_paths
        }
        
        if errors:
            response_data["warnings"] = errors
            
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Critical error in video upload: {e}")
        return jsonify({
            "success": False, 
            "message": f"Server error during upload: {str(e)}"
        }), 500


@app.route("/start_analysis", methods=["POST"])
def start_analysis():
    """Start video analysis with enhanced controls"""
    try:
        if system_state.analysis_running:
            return jsonify({
                "success": False, 
                "message": "Analysis is already running"
            }), 400
        
        if not system_state.video_paths:
            return jsonify({
                "success": False, 
                "message": "No videos uploaded. Please upload videos first."
            }), 400
        
        # Parse optional parameters from request
        data = request.get_json() or {}
        interval = data.get('interval', config.ANALYSIS_INTERVAL)
        max_iterations = data.get('max_iterations', config.MAX_ITERATIONS)
        
        # Validate parameters
        if not (1 <= interval <= 60):
            return jsonify({
                "success": False,
                "message": "Interval must be between 1 and 60 seconds"
            }), 400
            
        if not (1 <= max_iterations <= 100):
            return jsonify({
                "success": False,
                "message": "Max iterations must be between 1 and 100"
            }), 400
        
        # Reset state and start analysis
        system_state.stop_thread = False
        system_state.latest_result = {}
        system_state.results_history = []
        
        # Start analysis in background thread
        analysis_thread = threading.Thread(
            target=analyze_videos, 
            args=(interval, max_iterations),
            daemon=True
        )
        system_state.analysis_thread = analysis_thread
        analysis_thread.start()
        
        logger.info(f"Analysis started with {len(system_state.video_paths)} videos, "
                   f"{interval}s interval, {max_iterations} max iterations")
        
        return jsonify({
            "success": True,
            "message": f"Analysis started with {len(system_state.video_paths)} video(s)",
            "config": {
                "interval": interval,
                "max_iterations": max_iterations,
                "video_count": len(system_state.video_paths)
            }
        })
        
    except Exception as e:
        logger.error(f"Error starting analysis: {e}")
        return jsonify({
            "success": False,
            "message": f"Failed to start analysis: {str(e)}"
        }), 500


@app.route("/stop_analysis", methods=["POST"])
def stop_analysis():
    """Stop video analysis"""
    try:
        system_state.stop_thread = True
        
        # Wait for thread to finish (with timeout)
        if system_state.analysis_thread and system_state.analysis_thread.is_alive():
            system_state.analysis_thread.join(timeout=5.0)
        
        logger.info("Analysis stopped by user request")
        return jsonify({
            "success": True, 
            "message": "Analysis stopped successfully"
        })
        
    except Exception as e:
        logger.error(f"Error stopping analysis: {e}")
        return jsonify({
            "success": False,
            "message": f"Error stopping analysis: {str(e)}"
        }), 500


@app.route("/analysis_status")
def analysis_status():
    """Get comprehensive analysis status"""
    try:
        return jsonify({
            "running": system_state.analysis_running,
            "videos_loaded": len(system_state.video_paths),
            "has_results": bool(system_state.latest_result),
            "results_count": len(system_state.results_history),
            "last_update": system_state.latest_result.get("timestamp"),
            "current_iteration": system_state.latest_result.get("iteration", 0)
        })
    except Exception as e:
        logger.error(f"Error getting analysis status: {e}")
        return jsonify({"error": "Failed to get status"}), 500


@app.route("/latest_data")
def latest_data():
    """Get latest analysis results with error handling"""
    try:
        return jsonify(system_state.latest_result)
    except Exception as e:
        logger.error(f"Error getting latest data: {e}")
        return jsonify({"error": "Failed to get latest data"}), 500


@app.route("/results_history")
def results_history():
    """Get analysis results history"""
    try:
        return jsonify({
            "history": system_state.results_history,
            "total_count": len(system_state.results_history)
        })
    except Exception as e:
        logger.error(f"Error getting results history: {e}")
        return jsonify({"error": "Failed to get history"}), 500


@app.route("/static_results/<filename>")
def static_results(filename):
    """Serve analysis result images with better error handling.

    Uses absolute paths and adds cache-busting headers. Provides verbose logging
    to help diagnose missing image issues on the frontend.
    """
    try:
        secure_name = secure_filename(filename)
        search_folders = [config.STATIC_RESULTS_FOLDER, config.RESULTS_FOLDER]

        for folder in search_folders:
            file_path = Path(folder) / secure_name
            if file_path.exists() and file_path.is_file() and file_path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                try:
                    resp = send_file(str(file_path), mimetype="image/jpeg")
                    # Disable caching so the browser always fetches freshest frame
                    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
                    resp.headers['Pragma'] = 'no-cache'
                    resp.headers['Expires'] = '0'
                    logger.debug(f"Served image: {file_path}")
                    return resp
                except Exception as e:
                    logger.error(f"send_file failed for {file_path}: {e}")
                    break

        # If we reach here, not found
        existing = []
        for folder in search_folders:
            existing.extend([p.name for p in Path(folder).glob('lane*_orig.jpg')])
            existing.extend([p.name for p in Path(folder).glob('lane*.jpg')])
        logger.warning(f"Image not found: {secure_name}. Existing candidates: {existing}")
        return jsonify({"error": "Image not found", "requested": secure_name, "available": existing}), 404
    except Exception as e:
        logger.error(f"Error serving image {filename}: {e}")
        return jsonify({"error": "Failed to serve image"}), 500


@app.route("/config", methods=["GET", "POST"])
def configuration():
    """Get or update system configuration"""
    try:
        if request.method == "GET":
            return jsonify({
                "model": config.YOLO_MODEL,
                "target_classes": config.TARGET_CLASSES,
                "confidence_threshold": config.CONFIDENCE_THRESHOLD,
                "iou_threshold": config.IOU_THRESHOLD,
                "frame_width": config.FRAME_WIDTH,
                "frame_height": config.FRAME_HEIGHT,
                "analysis_interval": config.ANALYSIS_INTERVAL,
                "max_iterations": config.MAX_ITERATIONS,
                "min_green_time": config.MIN_GREEN_TIME,
                "max_green_time": config.MAX_GREEN_TIME,
                "default_cycle_time": config.DEFAULT_CYCLE_TIME
            })
        
        elif request.method == "POST":
            if system_state.analysis_running:
                return jsonify({
                    "success": False,
                    "message": "Cannot update config while analysis is running"
                }), 400
            
            data = request.get_json()
            updated_fields = []
            
            # Update configurable parameters
            if "confidence_threshold" in data:
                new_val = float(data["confidence_threshold"])
                if 0.1 <= new_val <= 1.0:
                    config.CONFIDENCE_THRESHOLD = new_val
                    updated_fields.append("confidence_threshold")
            
            if "iou_threshold" in data:
                new_val = float(data["iou_threshold"])
                if 0.1 <= new_val <= 1.0:
                    config.IOU_THRESHOLD = new_val
                    updated_fields.append("iou_threshold")
            
            if "analysis_interval" in data:
                new_val = int(data["analysis_interval"])
                if 1 <= new_val <= 60:
                    config.ANALYSIS_INTERVAL = new_val
                    updated_fields.append("analysis_interval")
            
            if "max_iterations" in data:
                new_val = int(data["max_iterations"])
                if 1 <= new_val <= 100:
                    config.MAX_ITERATIONS = new_val
                    updated_fields.append("max_iterations")
            
            return jsonify({
                "success": True,
                "message": f"Updated: {', '.join(updated_fields)}",
                "updated_fields": updated_fields
            })
            
    except Exception as e:
        logger.error(f"Error in configuration endpoint: {e}")
        return jsonify({"error": "Configuration error"}), 500


@app.route("/reset", methods=["POST"])
def reset_system():
    """Reset the entire system state"""
    try:
        # Stop any running analysis
        system_state.stop_thread = True
        if system_state.analysis_thread and system_state.analysis_thread.is_alive():
            system_state.analysis_thread.join(timeout=5.0)
        
        # Reset state
        system_state.reset()
        
        # Clear uploaded files (optional)
        clear_files = request.get_json().get("clear_files", False) if request.get_json() else False
        if clear_files:
            for folder in [config.UPLOAD_FOLDER, config.RESULTS_FOLDER, config.STATIC_RESULTS_FOLDER]:
                folder_path = Path(folder)
                if folder_path.exists():
                    for file_path in folder_path.glob("*"):
                        if file_path.is_file():
                            file_path.unlink()
        
        logger.info("System reset completed")
        return jsonify({
            "success": True,
            "message": "System reset successfully",
            "files_cleared": clear_files
        })
        
    except Exception as e:
        logger.error(f"Error resetting system: {e}")
        return jsonify({
            "success": False,
            "message": f"Reset failed: {str(e)}"
        }), 500


@app.route("/health")
def health():
    """Enhanced health check endpoint"""
    try:
        # Check system components
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "model_loaded": model is not None,
                "upload_folder": Path(config.UPLOAD_FOLDER).exists(),
                "results_folder": Path(config.RESULTS_FOLDER).exists(),
                "static_results_folder": Path(config.STATIC_RESULTS_FOLDER).exists(),
                "analysis_running": system_state.analysis_running,
                "videos_loaded": len(system_state.video_paths)
            },
            "config": {
                "yolo_model": config.YOLO_MODEL,
                "target_classes": config.TARGET_CLASSES,
                "confidence_threshold": config.CONFIDENCE_THRESHOLD
            }
        }
        
        # Check if any critical components are failing
        critical_components = ["model_loaded", "upload_folder", "results_folder"]
        if not all(health_status["components"][comp] for comp in critical_components):
            health_status["status"] = "unhealthy"
            return jsonify(health_status), 503
        
        return jsonify(health_status)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(413)
def too_large(error):
    return jsonify({"error": "File too large"}), 413


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500


# -------------------------
# Main application
# -------------------------
if __name__ == "__main__":
    try:
        logger.info("Starting Enhanced Traffic Management System...")
        logger.info(f"Upload folder: {Path(config.UPLOAD_FOLDER).absolute()}")
        logger.info(f"Results folder: {Path(config.RESULTS_FOLDER).absolute()}")
        logger.info(f"Static results folder: {Path(config.STATIC_RESULTS_FOLDER).absolute()}")
        logger.info(f"YOLO model: {config.YOLO_MODEL}")
        logger.info(f"Target classes: {config.TARGET_CLASSES}")
        
        # Run the Flask application
        app.run(
            debug=False,  # Set to False for production
            host="0.0.0.0",
            port=5000,
            threaded=True
        )
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    finally:
        # Cleanup on exit
        if system_state.analysis_running:
            system_state.stop_thread = True
            if system_state.analysis_thread:
                system_state.analysis_thread.join(timeout=10)
        logger.info("Application shutdown complete")