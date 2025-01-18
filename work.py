from PIL import Image
import numpy as np
from classifiers import Meso4

model = Meso4()
model.load('weights/Meso4_DF.h5')

image_path = "data/df/df00204.jpg"
img = Image.open(image_path).resize((256, 256)).convert('RGB')

# Preprocess the image
img_array = np.array(img)  # Convert image to a NumPy array
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array = img_array.astype('float32') / 255.0  # Normalize pixel values to [0, 1]


prediction = model.predict(img_array)
print(prediction)