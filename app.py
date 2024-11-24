from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import requests
import os
import tensorflow as tf
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps
import base64
from io import BytesIO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['UPLOAD_FOLDER'] = 'uploads/'
db = SQLAlchemy(app)

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Database model for user accounts
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

# Load the ML model
model = tf.keras.models.load_model('keras_model.h5')  # Changed to load .h5 format model

# Load labels from labels.txt
with open('labels.txt', 'r') as f:
    labels = [line.strip().split(' ', 1)[1] for line in f.readlines()]

# Home route
@app.route('/')
def home():
    return render_template('index.html', css_url=url_for('static', filename='css/style.css'))  # Homepage for Skindom

# User registration route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        hashed_password = generate_password_hash(password, method='sha256')

        new_user = User(email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful. Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', css_url=url_for('static', filename='css/style.css'))

# User login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            flash('Login successful.', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password', 'error')
            return redirect(url_for('login'))
    return render_template('login.html', css_url=url_for('static', filename='css/style.css'))

# User dashboard route
@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'user_id' not in session:
        flash('Please log in to access the dashboard.', 'error')
        return redirect(url_for('login'))

    if request.method == 'POST':
        try:
            # Handle image upload or camera capture
            if 'image' in request.files and request.files['image'].filename != '':
                file = request.files['image']
                # Save the uploaded file
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                # Load and preprocess the image
                image = Image.open(filepath).convert("RGB")
            elif 'camera_image' in request.form and request.form['camera_image'] != '':
                # Handle camera image (base64 string)
                camera_image_data = request.form['camera_image']
                # Remove the data:image/png;base64, part
                camera_image_data = camera_image_data.split(',')[1]
                image_data = base64.b64decode(camera_image_data)
                # Create a PIL Image from the decoded bytes
                image = Image.open(BytesIO(image_data)).convert("RGB")
            else:
                return jsonify({"success": False, "error": "No file selected"}), 400

            # Resize and crop the image to the required size (224x224)
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

            # Turn the image into a numpy array
            image_array = np.asarray(image)

            # Normalize the image
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

            # Prepare data to be fed to the model
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = normalized_image_array

            # Make the prediction
            prediction = model.predict(data)
            index = np.argmax(prediction)
            predicted_label = labels[index]
            confidence_score = prediction[0][index]

            # Provide recommendations based on prediction
            recommendations = {}
            if predicted_label == "Acne":
                recommendations = {
                    "do": [
                        "Dietary Adjustments: Avoid high-glycemic index foods, include foods rich in antioxidants and omega-3s.",
                        "Natural Remedies: Apply tea tree oil, aloe vera, or a turmeric mask.",
                        "Hygiene: Change pillowcases regularly, cleanse face twice daily.",
                        "Lifestyle: Manage stress and maintain consistent exercise."
                    ],
                    "dont": [
                        "Avoid touching or picking at your face.",
                        "Avoid oily hair or makeup products.",
                        "Reconsider shaving if it irritates your skin."
                    ]
                }
            elif predicted_label == "Scarring":
                recommendations = {
                    "do": [
                        "Natural Remedies: Apply aloe vera, honey, coconut oil, lemon juice, or vitamin E to scars.",
                        "Skincare: Gentle exfoliation, keep skin hydrated, use sun protection.",
                        "Lifestyle: Eat collagen-rich foods and drink plenty of water."
                    ],
                    "dont": [
                        "Avoid picking or scratching scars.",
                        "Avoid harsh scrubs.",
                        "Avoid excessive sun exposure."
                    ]
                }
            elif predicted_label == "Aging Skin":
                recommendations = {
                    "do": [
                        "Natural Remedies: Apply aloe vera, rosehip oil, avocado mask, and green tea.",
                        "Diet: Consume antioxidant-rich and collagen-boosting foods.",
                        "Lifestyle: Use sun protection, cleanse gently, and sleep regularly."
                    ],
                    "dont": [
                        "Avoid smoking.",
                        "Avoid excess sugar.",
                        "Avoid harsh products."
                    ]
                }
            elif predicted_label == "Dry Skin":
                recommendations = {
                    "do": [
                        "Natural Moisturizing Remedies: Apply coconut oil, aloe vera, honey, or olive oil.",
                        "Hydrating Skincare: Use a gentle cleanser and a rich moisturizer.",
                        "Lifestyle: Drink plenty of water and avoid hot showers."
                    ],
                    "dont": [
                        "Avoid excessive exfoliation.",
                        "Avoid harsh soaps and chemicals.",
                        "Avoid prolonged sun exposure."
                    ]
                }
            elif predicted_label == "Oily Skin":
                recommendations = {
                    "do": [
                        "Natural Remedies: Apply tea tree oil, witch hazel, aloe vera, or lemon juice.",
                        "Skincare: Use an oil-free cleanser, clay masks, and a non-comedogenic moisturizer.",
                        "Diet: Avoid high-glycemic foods, increase omega-3 intake."
                    ],
                    "dont": [
                        "Avoid over-washing.",
                        "Avoid heavy creams.",
                        "Avoid touching your face."
                    ]
                }
            elif predicted_label == "Normal Skin":
                recommendations = {
                    "do": [
                        "Natural Skincare: Apply aloe vera, use rose water, cucumber, and lightweight moisturizer.",
                        "Diet: Eat a balanced diet, consume omega-3s, and stay hydrated.",
                        "Lifestyle: Clean sleeping environment, gentle cleansing, and exercise."
                    ],
                    "dont": [
                        "Avoid harsh products.",
                        "Avoid over-washing.",
                        "Avoid sun damage."
                    ]
                }

            # Return the prediction result, confidence score, and recommendations as JSON
            return jsonify({
                "success": True,
                "result": predicted_label,
                "confidence": float(confidence_score),
                "recommendations": recommendations
            })

        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    return render_template('dashboard.html', css_url=url_for('static', filename='css/style.css'))

# Logout route
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('home'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
