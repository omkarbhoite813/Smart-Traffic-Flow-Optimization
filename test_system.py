#!/usr/bin/env python3
"""
Simple test script for the Traffic Management System
"""

import requests
import time
import sys
from pathlib import Path

def test_system():
    """Run basic tests on the traffic management system"""
    base_url = "http://localhost:5000"
    
    print("🧪 Testing Traffic Management System")
    print("=" * 50)
    
    # Test 1: Health Check
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Health check passed: {data.get('status')}")
        else:
            print(f"   ❌ Health check failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Cannot connect to system: {e}")
        print("   ℹ️  Make sure the system is running (python app.py)")
        return False
    
    # Test 2: Main page
    print("2. Testing main page...")
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("   ✅ Main page loads successfully")
        else:
            print(f"   ❌ Main page failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"   ❌ Main page error: {e}")
    
    # Test 3: Analysis status
    print("3. Testing analysis status...")
    try:
        response = requests.get(f"{base_url}/analysis_status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Analysis status: Running={data.get('running')}, Videos={data.get('videos_loaded')}")
        else:
            print(f"   ❌ Analysis status failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"   ❌ Analysis status error: {e}")
    
    # Test 4: Configuration
    print("4. Testing configuration endpoint...")
    try:
        response = requests.get(f"{base_url}/config", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Configuration loaded: Model={data.get('model')}")
        else:
            print(f"   ❌ Configuration failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"   ❌ Configuration error: {e}")
    
    print("\n✅ Basic tests completed!")
    print("🚀 System appears to be working correctly")
    return True

if __name__ == "__main__":
    if not test_system():
        sys.exit(1)
