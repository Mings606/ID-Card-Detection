import os
import subprocess
import sys

def install_requirements():
    """
    Install Python libraries from the requirements.txt file.
    """
    print("Installing Python dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Python dependencies installed successfully.")
    except Exception as e:
        print(f"Error installing Python dependencies: {e}")
        sys.exit(1)

def install_tesseract():
    """
    Install Tesseract-OCR on the user's system.
    """
    print("Checking for Tesseract installation...")
    try:
        if sys.platform == "darwin":  # macOS
            subprocess.check_call(["brew", "install", "tesseract"])
        elif sys.platform.startswith("linux"):
            subprocess.check_call(["sudo", "apt-get", "install", "-y", "tesseract-ocr"])
        elif os.name == "nt":  # Windows
            print("Please download and install Tesseract manually from https://github.com/UB-Mannheim/tesseract/wiki.")
            input("Press Enter after installation is complete.")
        print("Tesseract installed successfully.")
    except Exception as e:
        print(f"Error installing Tesseract: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("Setting up the environment for your TensorFlow script.")
    install_requirements()
    install_tesseract()
    print("Setup complete. You're ready to run the script!")
