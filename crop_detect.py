import cv2
from PIL import Image
import numpy as np
from classifiers import Meso4

# used for debugging purposes. Can remove it if you wish
# import matplotlib.pyplot as plt

# Load the pre-trained model
model = Meso4()
model.load('weights/Meso4_DF.h5')

def detect_and_crop_face_opencv(image_path):
    # Load Haar cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or unable to load.")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print("No faces detected!")
        return None

    # Crop the first detected face
    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]

    # Resize the face to 256x256
    face = cv2.resize(face, (256, 256))
    return Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

# Process the image
image_path = "data/real/real00240.jpg"
face = detect_and_crop_face_opencv(image_path)

if face:
    # # Display the cropped face
    # plt.imshow(face)
    # plt.axis("off")  # Hide axes for better visualization
    # plt.title("Detected Face")
    # plt.show()

    # Convert face to NumPy array and preprocess
    face_array = np.array(face) / 255.0  # Normalize
    face_array = np.expand_dims(face_array, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(face_array)
    print(prediction[0][0])
    # If value is > 0.5 it is real else fake or a deepfake
    # The values close to 0 or 1 really count else values have uncertainities
else:
    print("No face detected or an error occurred.")
