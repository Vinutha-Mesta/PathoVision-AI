import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

# Optional: for Grad-CAM plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Config
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'images', 'uploads')
HEATMAP_FOLDER = os.path.join(BASE_DIR, 'static', 'images', 'heatmaps')
MODEL_FOLDER = os.path.join(BASE_DIR, 'models')

ALLOWED_EXT = {'png', 'jpg', 'jpeg', 'tiff', 'bmp'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = 'e425cf21833cf49d2da80704091c9238'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 12 * 1024 * 1024  # 12 MB limit

# ========== Model loading ==========
tier1_model = None
colon_stage_model = None
lung_stage_model = None

def load_models():
    global tier1_model, colon_stage_model, lung_stage_model
    try:
        tier1_model = load_model(os.path.join(MODEL_FOLDER, 'resnet50_core_model_finetuned.h5'))
        colon_stage_model = load_model(os.path.join(MODEL_FOLDER, 'densenet121_model_colon.h5'))
        lung_stage_model = load_model(os.path.join(MODEL_FOLDER, 'densenet121_model_lung.h5'))
        print("Models loaded successfully.")
    except Exception as e:
        print("Model load warning:", e)

# Call at startup
load_models()

# ========== Utility functions ==========
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def preprocess_image_for_model(img_path, target_size=(224,224)):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(target_size)
    arr = np.array(img).astype('float32') / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# ========== Prediction placeholders ==========
def predict_tier1(img_path):
    val = hash(img_path) % 100
    organ = 'Colon' if val % 2 == 0 else 'Lung'
    status = 'Cancer' if val % 3 == 0 else 'Normal'
    confidence = 0.85
    return organ, status, confidence

def predict_stage(img_path, organ):
    val = (hash(img_path) % 4) + 1
    stage = f"Stage {val}"
    confidence = 0.78
    return stage, confidence

# ========== Grad-CAM placeholder ==========
def generate_gradcam_placeholder(img_path, model, save_to, target_size=(224,224)):
    try:
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Find the last convolutional layer for DenseNet121
        last_conv_layer_name = None
        for layer in reversed(model.layers):
            if len(layer.output_shape) == 4:  # 4D tensor -> Conv layer
                last_conv_layer_name = layer.name
                break

        if last_conv_layer_name is None:
            raise ValueError("No convolutional layer found in model.")

        # Create a Grad-CAM model that maps input image to activations and predictions
        grad_model = Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )

        # Record operations for automatic differentiation
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        # Compute gradients of predicted class wrt conv layer
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight activation maps with gradients
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)

        # Normalize the heatmap
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        heatmap = heatmap.numpy()

        # Resize heatmap to original image size
        img_orig = cv2.imread(img_path)
        heatmap = cv2.resize(heatmap, (img_orig.shape[1], img_orig.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Superimpose heatmap onto original image
        overlay = cv2.addWeighted(img_orig, 0.6, heatmap, 0.4, 0)
        cv2.imwrite(save_to, overlay)
        return True
    except Exception as e:
        print("Grad-CAM generation failed:", e)
        return False

# ========== Routes ==========
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/analyze', methods=['GET','POST'])
def analyze():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_name = f"{uuid.uuid4().hex}_{filename}"
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
            file.save(save_path)

            organ, status, conf1 = predict_tier1(save_path)

            stage = None
            conf_stage = None
            if status.lower() == 'cancer':
                stage, conf_stage = predict_stage(save_path, organ)

            heatmap_name = f"heatmap_{uuid.uuid4().hex}.png"
            heatmap_path = os.path.join(HEATMAP_FOLDER, heatmap_name)
            heat_ok = generate_gradcam_placeholder(save_path, heatmap_path)
            if not heat_ok:
                heatmap_name = None

            return render_template('results.html',
                                   organ=organ,
                                   status=status,
                                   confidence=round(conf1, 3),
                                   stage=stage,
                                   stage_confidence=round(conf_stage,3) if conf_stage else None,
                                   heatmap=('heatmaps/' + heatmap_name) if heatmap_name else None)
        else:
            flash('Unsupported file type. Allowed: png, jpg, jpeg, tiff, bmp', 'warning')
            return redirect(request.url)

    return render_template('analyze.html')

@app.route('/results')
def results():
    return render_template('results.html', organ=None, status=None, heatmap=None)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/methodology')
def methodology():
    return render_template('methodology.html')

@app.route('/contact', methods=['GET','POST'])
def contact():
    if request.method == 'POST':
        flash('Thanks — your message has been received.', 'success')
        return redirect(url_for('contact'))
    return render_template('contact.html')

@app.route('/disclaimer')
def disclaimer():
    return render_template('disclaimer.html')

if __name__ == '__main__':
    print("Starting Detectr+ Flask app...")
    print("Open in browser: http://127.0.0.1:5000/")
    app.run(host='0.0.0.0', port=5000, debug=True)