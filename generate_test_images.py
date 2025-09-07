#!/usr/bin/env python3
"""
Quick test script to generate test images and verify the system
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add the current directory to the path so we can import from app.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from app import create_test_frame_with_annotations, config
    print("✅ Successfully imported functions from app.py")
except ImportError as e:
    print(f"❌ Failed to import from app.py: {e}")
    # Fallback - create simple test images
    
    def create_simple_test_frame(lane_id):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Different colors for each lane
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
        color = colors[lane_id % 4]
        
        # Fill with color
        frame[:] = color
        
        # Add text
        cv2.putText(frame, f"TEST LANE {lane_id + 1}", (150, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
        
        # Add some rectangles to simulate vehicles
        cv2.rectangle(frame, (50, 100), (150, 200), (255, 255, 255), 2)
        cv2.rectangle(frame, (200, 300), (350, 400), (255, 255, 255), 2)
        
        return frame

def main():
    print("🧪 Generating test images for traffic management system...")
    
    # Create directories if they don't exist
    results_dir = Path("results")
    static_results_dir = Path("static_results")
    
    results_dir.mkdir(exist_ok=True)
    static_results_dir.mkdir(exist_ok=True)
    
    print(f"📁 Created directories: {results_dir}, {static_results_dir}")
    
    # Generate test images for each lane
    for i in range(4):
        try:
            # Try to use the proper function if available
            if 'create_test_frame_with_annotations' in globals():
                test_frame = create_test_frame_with_annotations(i)
            else:
                test_frame = create_simple_test_frame(i)
            
            # Save processed images
            lane_path = results_dir / f"lane{i+1}.jpg"
            static_lane_path = static_results_dir / f"lane{i+1}.jpg"
            
            cv2.imwrite(str(lane_path), test_frame)
            cv2.imwrite(str(static_lane_path), test_frame)
            
            # Create simple original frames
            orig_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            orig_frame[:] = (100, 150, 200)  # Light blue background
            cv2.putText(orig_frame, f"ORIGINAL LANE {i+1}", (100, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            
            orig_path = results_dir / f"lane{i+1}_orig.jpg"
            static_orig_path = static_results_dir / f"lane{i+1}_orig.jpg"
            
            cv2.imwrite(str(orig_path), orig_frame)
            cv2.imwrite(str(static_orig_path), orig_frame)
            
            print(f"✅ Generated images for lane {i+1}")
            
        except Exception as e:
            print(f"❌ Error generating images for lane {i+1}: {e}")
    
    # Verify files were created
    print("\n📋 Verifying created files:")
    for folder in [results_dir, static_results_dir]:
        print(f"\n{folder}:")
        files = list(folder.glob("*.jpg"))
        for file in sorted(files):
            size = file.stat().st_size
            print(f"  ✅ {file.name} ({size:,} bytes)")
    
    print(f"\n🎉 Test image generation complete!")
    print("You can now:")
    print("1. Start the server: python app.py")
    print("2. Open http://localhost:5000 in your browser")
    print("3. The test images should be visible in the frontend")

if __name__ == "__main__":
    main()
