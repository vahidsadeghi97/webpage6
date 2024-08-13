import os
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import torch
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

app = Flask(__name__)

# Model and Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_names = ['Scar', 'Normal', 'Stage 1', 'Stage 2', 'Stage 3']

# Load the trained model
model = models.vgg16(pretrained=False)
num_ftrs = model.classifier[6].in_features
model.classifier[6] = torch.nn.Linear(num_ftrs, len(class_names))
model.load_state_dict(torch.load('C:\\Users\\Yaran\\Desktop\\vgg16_weights.pth', map_location=device))
model.to(device)
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

UPLOAD_FOLDER = 'static/uploads/'
ENHANCED_FOLDER = 'static/enhanced/'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ENHANCED_FOLDER'] = ENHANCED_FOLDER

# Route for homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling image upload and classification
@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Process and classify the image
            img = Image.open(filepath)
            img = img.convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(img_tensor)
                _, preds = torch.max(outputs, 1)
                predicted_class = class_names[preds.item()]

            return render_template('index.html', filename=filename, prediction=predicted_class)
    return redirect(url_for('index'))

# Route for displaying the uploaded image
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Route for Grad-CAM++ visualization
@app.route('/gradcam/<filename>', methods=['GET'])
def gradcam(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img = Image.open(filepath)
    img = img.convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Initialize GradCAM++
    target_layers = [model.features[-1]]
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)

    # Generate Grad-CAM++ heatmap
    with torch.no_grad():
        outputs = model(img_tensor)
        _, preds = torch.max(outputs, 1)
        target_category = preds.item()

    targets = [ClassifierOutputTarget(target_category)]
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0]

    # Convert the image to numpy array and normalize
    input_image = np.array(img).astype(np.float32) / 255

    # Resize the heatmap to match the input image dimensions
    heatmap_resized = cv2.resize(grayscale_cam, (input_image.shape[1], input_image.shape[0]))

    # Overlay the resized heatmap on the input image
    visualization = show_cam_on_image(input_image, heatmap_resized, use_rgb=True)

    # Convert visualization to PNG and encode to base64
    buffered = BytesIO()
    plt.imsave(buffered, visualization, format="png")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return render_template('index.html', filename=filename, gradcam_img=img_str)

# Helper function to check allowed file extensions
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create directory for uploads if it doesn't exist
    os.makedirs(ENHANCED_FOLDER, exist_ok=True)  # Create directory for enhanced images if it doesn't exist
    app.run(host='0.0.0.0', port=5000)
