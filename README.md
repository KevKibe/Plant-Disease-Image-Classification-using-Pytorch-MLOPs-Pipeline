## Problem Statement
This is a project to build aa ResNet CNN model that can be used to classify images of plant leaves as healthy or diseased. The model can also identify the specific disease that a leaf is suffering from, if applicable. The repository also includes a link to the training data that was used to train the model, as well as instructions on how to use the model.

## Dataset
The dataset is from [Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) consisting of 87k images of healthy and diseased crop leaves categorized into 38 classes. The dataset is divided into 80/20 ratio of training and validation sets and a test set of 33 images.

## Training and Deployment
The model was [trained](https://github.com/KevKibe/Plant-Disease-Image-Classification-using-Pytorch/blob/main/plant-disease-classification-resnet19.ipynb) on a custom ResNet9 CNN architecture and deployed as a [REST API](https://github.com/KevKibe/Plant-Disease-Image-Classification-using-Pytorch/blob/main/main.py) using Flask through a Docker image to Google Cloud Run. 

## Getting Started
- Clone the repository: `git clone https://github.com/KevKibe/KevKibe/Plant-Disease-Image-Classification-using-Pytorch`
- Navigate to the project directory: `cd Plant-Disease-Image-Classification-using-Pytorch`
- Install the dependencies: `pip install -r requirements.txt`

## Usage
- Start the Flask API: `python main.py`
- Access the web interface at: `http://localhost:5000`
- Run the model on test data: `python test.py`
- You can run tests on the images in the [Test File](https://github.com/KevKibe/Plant-Disease-Image-Classification-using-Pytorch/tree/main/test)
  
