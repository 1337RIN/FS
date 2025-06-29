from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import os
from werkzeug.utils import secure_filename
import logging
from model import UNet
from model1 import PlantDiseaseClassifier
import json
import base64
import io
import matplotlib.pyplot as plt

app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Define class labels (from predict1.py)
CLASS_LABELS = [
    "Парша яблони", "Чёрная гниль яблони", "Ржавчина яблони", "Здоровый яблоневый лист",
    "Здоровый лист черники", "Мучнистая роса вишни", "Здоровый лист вишни",
    "Пятнистость листьев кукурузы", "Обыкновенная ржавчина кукурузы",
    "Северная пятнистость листьев кукурузы", "Здоровый лист кукурузы",
    "Чёрная гниль винограда", "Эска винограда", "Пятнистость листьев винограда",
    "Здоровый лист винограда", "Зеленение цитрусовых", "Бактериальная пятнистость персика",
    "Здоровый лист персика", "Бактериальная пятнистость сладкого перца",
    "Здоровый лист сладкого перца", "Ранняя пятнистость картофеля",
    "Поздняя пятнистость картофеля", "Здоровый лист картофеля", "Здоровый лист малины",
    "Здоровый лист сои", "Мучнистая роса кабачка", "Ожог листьев клубники",
    "Здоровый лист клубники", "Бактериальная пятнистость томата",
    "Ранняя пятнистость томата", "Поздняя пятнистость томата", "Мучнистая роса томата",
    "Пятнистость листьев томата", "Паутинный клещ томата", "Пятнистость томата",
    "Вирус скручивания листьев томата", "Вирус мозаики томата", "Здоровый лист томата"
]

# Load models
try:
    # U-Net model (set n_classes=1 to match saved model)
    unet_model = UNet(n_channels=3, n_classes=1)
    unet_model.load_state_dict(torch.load('saved_models/model_epoch_9.pt', map_location=torch.device('cpu')))
    unet_model.eval()
    
    # ResNet18 model
    resnet_model = PlantDiseaseClassifier(num_classes=config['num_classes'])
    resnet_model.load_state_dict(torch.load('saved_models/best_model.pth', map_location=torch.device('cpu')))
    resnet_model.eval()
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    raise

# Image preprocessing (based on predict.py and predict1.py)
preprocess = transforms.Compose([
    transforms.Resize(tuple(config["input_size"])),  # 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    """Preprocess image for model input."""
    return preprocess(image).unsqueeze(0)  # Add batch dimension

def generate_segmentation_image(probability_map):
    """Convert probability map to a base64-encoded image."""
    plt.figure(figsize=(5, 5))
    plt.imshow(probability_map, cmap='gray')
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_base64

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload and run disease detection."""
    try:
        # Check if an image was uploaded
        if 'file' not in request.files:
            logger.error("No file part in request")
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        
        # Check if a file was selected
        if file.filename == '':
            logger.error("No selected file")
            return jsonify({'error': 'No selected file'}), 400
        
        # Validate file extension
        if not allowed_file(file.filename):
            logger.error("Invalid file extension")
            return jsonify({'error': 'Invalid file type. Only PNG, JPG, JPEG allowed'}), 400

        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        logger.info(f"File saved: {file_path}")

        # Open and preprocess the image
        image = Image.open(file_path).convert('RGB')
        image_tensor = preprocess_image(image)

        # Run inference on U-Net model
        with torch.no_grad():
            unet_output = unet_model(image_tensor)
            probability_map = torch.sigmoid(unet_output).squeeze().detach().numpy()
            # Threshold at 0.5 to detect diseased areas
            binary_mask = (probability_map > 0.5).astype(np.uint8)
            # Check if any disease is detected
            disease_detected = np.any(binary_mask)

        # Run inference on ResNet18 model
        with torch.no_grad():
            resnet_output = resnet_model(image_tensor)
            probabilities = torch.softmax(resnet_output, dim=1)[0]
            top_prob, top_class = torch.topk(probabilities, k=1)
            predicted_class = CLASS_LABELS[top_class.item()]
            confidence = float(top_prob.item())

        # Clean up uploaded file
        os.remove(file_path)
        logger.info(f"File removed: {file_path}")

        # Generate segmentation image as base64
        segmentation_image = generate_segmentation_image(probability_map)

        # Return results as JSON
        return jsonify({
            'status': 'success',
            'disease': predicted_class if disease_detected else "Здоровый лист",
            'confidence': confidence if disease_detected else 1.0,
            'segmentation_image': f"data:image/png;base64,{segmentation_image}"
        })

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)