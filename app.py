
from flask import Flask, request, render_template, send_from_directory, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import VGG16
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def create_model(input_shape, num_classes):
    vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in vgg_base.layers:
        layer.trainable = False

    x = Flatten()(vgg_base.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=vgg_base.input, outputs=predictions)
    return model

model_path = 'C:\Aswathy L\DISEASEFinal\VGG_PVG.h5'
input_shape = (224, 224, 3)
num_classes = 15

# Recreate the exact same model, including its weights and optimizer
model = create_model(input_shape, num_classes)
model.load_weights(model_path)  # Load weights

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = img_batch / 255.0

    prediction = model.predict(img_preprocessed)
    return prediction

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(file_path)
        preds = model_predict(file_path, model)

        # Debug: Print predictions to verify
        print(f"Predictions: {preds}")

        # Convert prediction to category name if needed
        pred_class = preds.argmax(axis=-1)  # Simple argmax

        # Debug: Print the predicted class index
        print(f"Predicted class index: {pred_class}")

        plant_disease_list = [
            "Not Valid",
            "Pepper,_bell___Bacterial_spot",
            "Pepper,_bell___healthy",
            "Potato___Early_blight",
            "Potato___Late_blight",
            "Potato___healthy",
            "Tomato___Bacterial_spot",
            "Tomato___Early_blight",
            "Tomato___Late_blight",
            "Tomato___Leaf_Mold",
            "Tomato___Septoria_leaf_spot",
            "Tomato___Spider_mites_Two-spotted_spider_mite",
            "Tomato___Target_Spot",
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
            "Tomato___Tomato_mosaic_virus",
            "Tomato___healthy"
        ]

        if 0 <= pred_class[0] < len(plant_disease_list):
            result = plant_disease_list[pred_class[0]]
        else:
            result = "Unknown class"

        filename = secure_filename(file.filename)
        return render_template('result.html', img_filename=filename, disease_name=result)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=5002)
