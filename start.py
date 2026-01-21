#!/usr/bin/env python3
"""
Wildlife Monitoring System - Startup Script

This script helps you start the wildlife monitoring web application.
Make sure you have installed all dependencies before running this script.

Usage:
    python start.py
"""

import os
import sys
import subprocess
import webbrowser
from pathlib import Path

def check_requirements():
    """Check if required directories and files exist"""
    print("🔍 Checking requirements...")

    # Check backend directory
    backend_dir = Path("backend")
    if not backend_dir.exists():
        print("❌ Backend directory not found!")
        return False

    # Check frontend directory
    frontend_dir = Path("frontend")
    if not frontend_dir.exists():
        print("❌ Frontend directory not found!")
        return False

    # Check models directory
    models_dir = backend_dir / "models"
    if not models_dir.exists():
        print("⚠️  Models directory not found. Creating it...")
        models_dir.mkdir(exist_ok=True)
        print("📁 Created models directory. Please place your model files there.")

    # Check model files (warn if missing)
    model_files = [
        "md_v5a.0.0.pt",
        "stage2_species_finetuned.pth",
        "stage3_autoencoder.pth"
    ]

    missing_models = []
    for model_file in model_files:
        if not (models_dir / model_file).exists():
            missing_models.append(model_file)

    if missing_models:
        print("⚠️  Missing model files:")
        for model in missing_models:
            print(f"   - {model}")
        print("📝 Please download and place the model files in backend/models/")
        print("🔗 The application will still start but predictions will fail.")

    return True

def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing dependencies...")

    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r",
            "backend/requirements.txt"
        ])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def start_server():
    """Start the Flask development server"""
    print("🚀 Starting Wildlife Monitoring System...")
    print("🌐 Frontend will be available at: http://localhost:5000")
    print("⏹️  Press Ctrl+C to stop the server")

    try:
        # Change to backend directory
        os.chdir("backend")

        # Start the Flask app
        subprocess.run([sys.executable, "app.py"])

    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")

def main():
    """Main startup function"""
    print("🦌 Wildlife Monitoring System")
    print("=" * 40)

    # Check if we're in the right directory
    if not Path("backend").exists() or not Path("frontend").exists():
        print("❌ Please run this script from the wildlife-monitoring root directory")
        sys.exit(1)

    # Check requirements
    if not check_requirements():
        sys.exit(1)

    # Install dependencies
    if not install_dependencies():
        print("❌ Please fix dependency issues and try again")
        sys.exit(1)

    # Start the server
    start_server()

if __name__ == "__main__":
    main()