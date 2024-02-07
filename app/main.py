import os
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from flask import Flask, request, jsonify
from PIL import Image
from flasgger import Swagger

app = Flask(__name__)
swagger = Swagger(app)


base_dir = os.path.dirname(os.path.abspath(__file__))

def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, in_channels, num_diseases):
        super(ResNet9, self).__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                         nn.Flatten(),
                                         nn.Linear(512, num_diseases))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

model = ResNet9(in_channels=3, num_diseases=38) 
model_path = os.path.join(base_dir, '../models/plant_disease_model.pth')
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

classnames_path = os.path.join(base_dir, '../app/class_names.txt')
class_names = open(classnames_path, 'r').read().splitlines()

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
def predict_image(img, model):
    xb = to_device(img.unsqueeze(0), device)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return class_names[preds[0].item()]

@staticmethod
@app.route('/', methods=['GET'])
def home():
    """
    Hello World endpoint.

    ---
    tags:
      - Home
    responses:
      200:
        description: Hello, World!
    """
    return "Hello, World!"


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict Plant Disease
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
    responses:
      200:
        description: Prediction
    """
    file = request.files['file']
    img = Image.open(file.stream)
    img_tensor = ToTensor()(img)
    prediction = predict_image(img_tensor, model)
    return {"prediction": prediction}

if __name__ == '__main__':
    app.run(debug= True, host='0.0.0.0', port=80)