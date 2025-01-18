import os
import cv2
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify
from flask_mail import Mail, Message
from werkzeug.utils import secure_filename
from classifiers import Meso4
from flask_cors import CORS

import logging

# Initialize Flask app
app = Flask(__name__)
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = 'subyyal12345@gmail.com'
app.config['MAIL_PASSWORD'] = 'nyel aehj ogqs hokj'
app.config['MAIL_DEFAULT_SENDER'] = 'awais910ax@gmail.com'

CORS(app);
mail = Mail(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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
        return None

    # Crop the first detected face
    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]

    # Resize the face to 256x256
    face = cv2.resize(face, (256, 256))
    return Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

@app.route('/detect', methods=['POST'])
def detect():
    
    logging.info("Request received for image detection")
    if 'image' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        # Detect and crop the face
        face = detect_and_crop_face_opencv(file_path)
        if face is None:
            return jsonify({"error": "No face detected"}), 400

        # Preprocess the face for the model
        face_array = np.array(face) / 255.0  # Normalize
        face_array = np.expand_dims(face_array, axis=0)  # Add batch dimension

        # Predict
        prediction = model.predict(face_array)
        result = "Real" if prediction[0][0] > 0.5 else "Fake"
        return jsonify({"result": result, "confidence": float(prediction[0][0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/send-email', methods=['POST'])
def send_email():
    data = request.json
    try:
        # Extract data from the request
        first_name = data['firstName']
        last_name = data['lastName']
        email = data['email']
        message_content = data['message']

        # Create the email message
        message = Message(
            subject="Thank You for Contacting Us!",
            recipients=[email],
            body=f"""
            Dear {first_name},
            
            Thank you for reaching out to us. Here is a copy of your message:
            
            "{message_content}"
            
            We will get back to you shortly.
            
            Best regards,
            Your Company
            """
        );

        # Send the email
        mail.send(message);
        
        message = Message(
            subject="New Contact Form Submission",
            recipients=['bajwa.saad1122@gmail.com'],
            body=f"""
            You have received a new message from the contact form Image Detection Website.
            
            First Name: {first_name}
            Last Name: {last_name}
            Email: {email}
            
            Message:
            "{message_content}"
            
            Best regards,
            Your Image Detection Website
            """
        )
        mail.send(message);

        return jsonify({'message': 'Email sent successfully!'}), 200
    except Exception as e: 
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)
