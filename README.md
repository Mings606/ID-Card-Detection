# TensorFlow ID Card Detection Script

## Setup Guide

1. **Install Python**: Ensure you have Python 3.8 or later installed.
   - Download from [https://www.python.org/](https://www.python.org/).

2. **Install Required Dependencies**:
   - Clone this repository or download the script.
   - Open a terminal and navigate to the project directory.
   - Run the following commands:

     ```bash
     pip install --upgrade pip
     python3 setup_environment.py
     ```

3. **Install Tesseract-OCR**:
   - The `setup_environment.py` script attempts to install Tesseract for macOS and Linux.
   - For Windows, download it manually from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki) and add the installation path to the system `PATH`.

4. **Run the Script**:
   - Connect your camera and execute the following:

     ```bash
     python3 id_card_detection.py
     ```

5. **Dependencies**:
   - TensorFlow
   - OpenCV
   - Pytesseract
   - NumPy

## Notes
- Ensure the TensorFlow model path and Tesseract path in your script are configured correctly for your system.
- For macOS users, install Homebrew if not already installed: [https://brew.sh/](https://brew.sh/).
