# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from visualizer import display_batch_of_images_with_gestures_and_hand_landmarks,annotate_image_with_gesture_and_landmarks
import cv2
# STEP 2: Create an GestureRecognizer object.
base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

# Open the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    # Convert the frame to an mp.Image
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # Recognize gestures in the frame
    recognition_result = recognizer.recognize(image)
    
    # Process the result
    if recognition_result.gestures:
        top_gesture = recognition_result.gestures[0][0]
        hand_landmarks = recognition_result.hand_landmarks
        result = (top_gesture, hand_landmarks)
        
        # Annotate the frame with the recognized gesture and hand landmarks
        annotated_image, gesture_name = annotate_image_with_gesture_and_landmarks(image, result)
    else:
        annotated_image = frame  # If no gestures are recognized, use the original frame
    
    # Display the annotated frame
    cv2.imshow('Gesture Recognition', annotated_image)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()