import pickle
from flask import Flask, request, jsonify
import torch
from torchvision.transforms import ToTensor
#from torchvision.models.resnet import resnet50
from torchvision.models.resnet import Resnet9
#from resnet9 import Resnet9

app = Flask(__name__)

model = torch.load('plant_disease_model.pkl')
model.eval()
class_names = open('class_names.txt', 'r').read().splitlines()

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict_image(img, model):
    """Converts image to array and return the predicted class
        with highest probability"""
    xb = to_device(img.unsqueeze(0), device)
    yb = model(xb)
    _, preds  = torch.max(yb, dim=1)

    return class_names[preds[0].item()]

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img = ToTensor()(file)
        prediction = predict_image(img, model)
        return prediction
    
if __name__ == '__main__':
    app.run(debug=True)