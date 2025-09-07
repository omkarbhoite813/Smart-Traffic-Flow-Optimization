#!/usr/bin/env python3
"""
Traffic Management System Utility Script
Provides system management, testing, and maintenance functions.
"""

import os
import sys
import json
import subprocess
import requests
import time
from pathlib import Path
from typing import Dict, List

class TrafficSystemManager:
    """Utility class for managing the traffic system"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.config_file = self.base_dir / "config.json"
        self.requirements_file = self.base_dir / "requirements.txt"
        self.app_file = self.base_dir / "app.py"
        
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are installed"""
        print("🔍 Checking dependencies...")
        
        try:
            import flask
            import cv2
            import numpy
            import ultralytics
            print("✅ All core dependencies found")
            return True
        except ImportError as e:
            print(f"❌ Missing dependency: {e}")
            return False
    
    def install_dependencies(self) -> bool:
        """Install required dependencies"""
        print("📦 Installing dependencies...")
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(self.requirements_file)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Dependencies installed successfully")
                return True
            else:
                print(f"❌ Installation failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Installation error: {e}")
            return False
    
    def check_yolo_models(self) -> Dict[str, bool]:
        """Check which YOLO models are available"""
        print("🤖 Checking YOLO models...")
        
        models = {
            'yolov8n.pt': self.base_dir / 'yolov8n.pt',
            'yolov8s.pt': self.base_dir / 'yolov8s.pt', 
            'yolov8m.pt': self.base_dir / 'yolov8m.pt',
            'yolov8l.pt': self.base_dir / 'yolov8l.pt',
            'yolov8x.pt': self.base_dir / 'yolov8x.pt'
        }
        
        status = {}
        for name, path in models.items():
            exists = path.exists()
            status[name] = exists
            print(f"{'✅' if exists else '❌'} {name}")
            
        return status
    
    def download_yolo_model(self, model_name: str = "yolov8s.pt") -> bool:
        """Download a YOLO model"""
        print(f"⬇️ Downloading {model_name}...")
        
        try:
            from ultralytics import YOLO
            model = YOLO(model_name)  # This will download the model
            print(f"✅ {model_name} downloaded successfully")
            return True
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False
    
    def test_system_health(self, host: str = "localhost", port: int = 5000) -> bool:
        """Test if the system is running and healthy"""
        print("🏥 Testing system health...")
        
        try:
            response = requests.get(f"http://{host}:{port}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ System is {data.get('status', 'unknown')}")
                
                # Check components
                components = data.get('components', {})
                for name, status in components.items():
                    print(f"  {'✅' if status else '❌'} {name}")
                
                return data.get('status') == 'healthy'
            else:
                print(f"❌ Health check failed: HTTP {response.status_code}")
                return False
                
        except requests.RequestException as e:
            print(f"❌ Cannot connect to system: {e}")
            return False
    
    def clean_directories(self) -> None:
        """Clean up temporary files and directories"""
        print("🧹 Cleaning directories...")
        
        directories_to_clean = ['uploads', 'results', 'static_results']
        
        for dir_name in directories_to_clean:
            dir_path = self.base_dir / dir_name
            if dir_path.exists():
                file_count = 0
                for file_path in dir_path.glob("*"):
                    if file_path.is_file():
                        try:
                            file_path.unlink()
                            file_count += 1
                        except Exception as e:
                            print(f"⚠️ Could not delete {file_path}: {e}")
                
                print(f"✅ Cleaned {dir_name}: {file_count} files removed")
            else:
                print(f"ℹ️ Directory {dir_name} does not exist")
    
    def create_sample_config(self) -> None:
        """Create a sample configuration file"""
        print("⚙️ Creating sample configuration...")
        
        sample_config = {
            "system": {
                "upload_folder": "uploads",
                "results_folder": "results",
                "static_results_folder": "static_results",
                "max_file_size_mb": 100,
                "allowed_extensions": ["mp4", "avi", "mov", "mkv", "wmv"]
            },
            "yolo": {
                "model": "yolov8s.pt",
                "target_classes": ["car", "truck", "motorcycle", "bus", "bicycle"],
                "confidence_threshold": 0.4,
                "iou_threshold": 0.5,
                "frame_width": 640,
                "frame_height": 480
            },
            "analysis": {
                "interval_seconds": 5,
                "max_iterations": 20,
                "min_green_time": 5.0,
                "max_green_time": 60.0,
                "default_cycle_time": 60.0
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(sample_config, f, indent=2)
        
        print(f"✅ Configuration created: {self.config_file}")
    
    def run_system(self, debug: bool = False) -> None:
        """Start the traffic management system"""
        print("🚀 Starting Traffic Management System...")
        
        try:
            env = os.environ.copy()
            if debug:
                env['FLASK_DEBUG'] = '1'
            
            subprocess.run([sys.executable, str(self.app_file)], env=env)
        except KeyboardInterrupt:
            print("\n🛑 System stopped by user")
        except Exception as e:
            print(f"❌ Error starting system: {e}")

def main():
    """Main CLI interface"""
    manager = TrafficSystemManager()
    
    if len(sys.argv) < 2:
        print("""
🚦 Traffic Management System Utility

Usage: python utils.py <command>

Commands:
  check-deps     - Check if dependencies are installed
  install-deps   - Install required dependencies
  check-models   - Check available YOLO models
  download-model - Download default YOLO model
  health-check   - Test if system is running
  clean          - Clean temporary files
  setup          - Complete system setup
  run            - Start the system
  run-debug      - Start system in debug mode

Examples:
  python utils.py setup        # Complete setup
  python utils.py health-check # Check system health
  python utils.py clean        # Clean files
        """)
        return
    
    command = sys.argv[1].lower()
    
    if command == "check-deps":
        manager.check_dependencies()
    
    elif command == "install-deps":
        manager.install_dependencies()
    
    elif command == "check-models":
        manager.check_yolo_models()
    
    elif command == "download-model":
        model = sys.argv[2] if len(sys.argv) > 2 else "yolov8s.pt"
        manager.download_yolo_model(model)
    
    elif command == "health-check":
        manager.test_system_health()
    
    elif command == "clean":
        manager.clean_directories()
    
    elif command == "setup":
        print("🔧 Setting up Traffic Management System...\n")
        
        # Check and install dependencies
        if not manager.check_dependencies():
            if not manager.install_dependencies():
                print("❌ Setup failed: Could not install dependencies")
                return
        
        # Check and download YOLO model
        models = manager.check_yolo_models()
        if not any(models.values()):
            if not manager.download_yolo_model():
                print("❌ Setup failed: Could not download YOLO model")
                return
        
        # Create config if it doesn't exist
        if not manager.config_file.exists():
            manager.create_sample_config()
        
        print("\n✅ Setup completed successfully!")
        print("Run 'python utils.py run' to start the system")
    
    elif command == "run":
        manager.run_system(debug=False)
    
    elif command == "run-debug":
        manager.run_system(debug=True)
    
    else:
        print(f"❌ Unknown command: {command}")
        print("Run 'python utils.py' for usage information")

if __name__ == "__main__":
    main()
