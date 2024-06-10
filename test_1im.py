# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from visualizer import annotate_image_with_gesture_and_landmarks 
import cv2
# STEP 2: Create an GestureRecognizer object.
base_options = python.BaseOptions(model_asset_path='exported_model/gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

IMAGE_FILENAMES = ['data_test/highfive.jpg']
for image_file_name in IMAGE_FILENAMES:
  # STEP 3: Load the input image.
  image = mp.Image.create_from_file(image_file_name)

  # STEP 4: Recognize gestures in the input image.
  recognition_result = recognizer.recognize(image)

  # STEP 5: Process the result. In this case, visualize it.
  top_gesture = recognition_result.gestures[0][0]
  hand_landmarks = recognition_result.hand_landmarks
  result=((top_gesture, hand_landmarks))

annotated_image,reg_ges = annotate_image_with_gesture_and_landmarks(image, result)
# Convert the annotated image to a format that can be displayed with OpenCV
annotated_image_cv = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

# Show the image with OpenCV
print(reg_ges)
cv2.imshow('Annotated Image', annotated_image_cv)
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()