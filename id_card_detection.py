import tensorflow as tf
import cv2
import os
import datetime
import pytesseract

# Specify the path to Tesseract-OCR
pytesseract.pytesseract.tesseract_cmd = r"/usr/local/bin/tesseract"  # Update with your Tesseract path

# Get the current working directory
BASE_DIR = os.getcwd()

# Set the model path relative to the project directory
MODEL_PATH = os.path.join(BASE_DIR, 'efficientdet_d4_coco17_tpu-32', 'saved_model')

# Set the output folder path
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'cropped_images')
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load TensorFlow model
print("Loading TensorFlow model...")
detect_fn = tf.saved_model.load(MODEL_PATH)
print("Model loaded successfully.")

def contains_required_text(cropped_image):
    """
    Check if the cropped image contains the required text.
    """
    text = pytesseract.image_to_string(cropped_image, lang="eng")
    required_keywords = ["HONG KONG"]
    return all(keyword in text for keyword in required_keywords), text


def detect_and_crop_id_card(frame, output_folder):
    """
    Detect and crop ID cards from a camera frame using TensorFlow model.
    """
    # Convert image to TensorFlow format
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Perform detection
    detections = detect_fn(input_tensor)

    # Extract detection data
    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()

    # Threshold to filter detections
    detection_threshold = 0.5
    id_card_detected = False

    for i in range(len(scores)):
        if scores[i] > detection_threshold:
            box = boxes[i]
            y_min, x_min, y_max, x_max = box
            cropped_id_card = frame[
                int(y_min * frame.shape[0]):int(y_max * frame.shape[0]),
                int(x_min * frame.shape[1]):int(x_max * frame.shape[1])
            ]

            if cropped_id_card.size > 0:
                contains_text, detected_text = contains_required_text(cropped_id_card)

                if contains_text:
                    # Save cropped image
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
                    cropped_filename = os.path.join(output_folder, f"final_crop_{timestamp}.png")
                    cv2.imwrite(cropped_filename, cropped_id_card)
                    print(f"Final crop saved: {cropped_filename}")
                    id_card_detected = True
                    break

    return id_card_detected


def process_camera():
    """
    Process frames from the camera and detect ID cards in real-time.
    """
    cap = cv2.VideoCapture(0)  # Open default camera
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    print("Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break

        # Detect ID card in the frame
        if detect_and_crop_id_card(frame, OUTPUT_FOLDER):
            print("ID card detected and cropped!")

        # Show the live feed
        cv2.imshow("Live Feed", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    process_camera()
