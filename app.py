from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io
from torchvision.models import resnet18
import torch.nn as nn

# Load the model
# model = torch.load('model_ft.pth')
model = resnet18(weights='ResNet18_Weights.DEFAULT')
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3) 
model.load_state_dict(torch.load('model_ft.pth'))
model.eval()

# define Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # get the image
    image_file = request.files['image'].read()
    image = Image.open(io.BytesIO(image_file))

    # preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image_tensor = preprocess(image)
    image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():
        model.eval()
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        labels = ['chair', 'sofa', 'bed']
        result = labels[predicted[0].item()]


    # return prediction result
    return jsonify({'result': result})

@app.route("/")
def index():
    return "Success!"

if __name__ == '__main__':
    app.run(host='0.0.0.0')