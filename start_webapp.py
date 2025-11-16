"""
Start the Hostel Price Prediction Web Application
Run this script to launch the backend server and access the web interface
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def print_banner():
    """Print startup banner"""
    print("\n" + "="*70)
    print("🏨 HOSTEL PRICE PREDICTION - WEB APPLICATION")
    print("="*70 + "\n")

def check_requirements():
    """Check if required packages are installed"""
    print("📦 Checking requirements...")
    
    required = ['flask', 'flask_cors', 'joblib', 'pandas', 'numpy', 'sklearn']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    print("✅ All requirements installed\n")
    return True

def check_model():
    """Check if trained model exists"""
    print("🤖 Checking for trained model...")
    
    base_dir = Path(__file__).parent
    model_path = base_dir / 'hostel_price_prediction' / 'models' / 'best_model.pkl'
    
    if not model_path.exists():
        print(f"\n❌ Model not found at: {model_path}")
        print("Please run the training pipeline first: python run_pipeline.py")
        return False
    
    print(f"✅ Model found at: {model_path}\n")
    return True

def start_server():
    """Start the Flask backend server"""
    print("🚀 Starting Flask backend server...\n")
    
    base_dir = Path(__file__).parent
    backend_script = base_dir / 'backend' / 'app.py'
    
    if not backend_script.exists():
        print(f"❌ Backend script not found at: {backend_script}")
        return None
    
    # Start the Flask server as a subprocess
    try:
        process = subprocess.Popen(
            [sys.executable, str(backend_script)],
            cwd=str(base_dir)
        )
        return process
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return None

def main():
    """Main function to start the application"""
    print_banner()
    
    # Check requirements
    if not check_requirements():
        return
    
    # Check model
    if not check_model():
        print("\n💡 To train the model, run:")
        print("   python run_pipeline.py")
        return
    
    # Start server
    server_process = start_server()
    
    if not server_process:
        return
    
    # Wait for server to start
    print("⏳ Waiting for server to start...")
    time.sleep(3)
    
    # Print access information
    print("\n" + "="*70)
    print("✅ WEB APPLICATION IS RUNNING!")
    print("="*70)
    print("\n📍 Access the application at:")
    print("   🌐 http://localhost:5000")
    print("\n📍 API endpoints:")
    print("   • Health Check: http://localhost:5000/api/health")
    print("   • Predict: http://localhost:5000/api/predict (POST)")
    print("   • Features: http://localhost:5000/api/features")
    print("\n💡 Press Ctrl+C to stop the server")
    print("="*70 + "\n")
    
    # Open browser
    try:
        webbrowser.open('http://localhost:5000')
        print("🌐 Opening web browser...\n")
    except:
        pass
    
    # Keep running
    try:
        server_process.wait()
    except KeyboardInterrupt:
        print("\n\n⚠️  Shutting down server...")
        server_process.terminate()
        server_process.wait()
        print("✅ Server stopped. Goodbye!\n")

if __name__ == "__main__":
    main()
