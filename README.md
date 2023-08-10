## Description
This repository contains the code implementation of a ResNet CNN model used to classify images of plant leaves as healthy or diseased. The model also identifies the specific disease that a leaf is suffering from, if applicable. The repository also includes a link to the training data that was used to train the model, as well as instructions on how to use the model.

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
  
## Deploying and Containerizing Your Application with Docker

Before you start, make sure you have [Docker](https://www.docker.com/get-started) installed on your system. 

1. **Clone the Repository:** First, clone the repository for your application to your local machine or cloud instance using the following commands:
   ```sh
   git clone https://github.com/KevKibe/Plant-Disease-Image-Classification-using-Pytorch.git
   cd Plant-Disease-Image-Classification-using-Pytorch
2.**Build the Docker Image:** Replace your-app-name with a suitable name for your application.
   ```
   docker build -t your-app-name .

 ```
   



## To deploy on an AWS EC2 instance
- Setup an EC2 instance and SSH to the instance.Use this as a [guide](https://www.machinelearningplus.com/deployment/deploy-ml-model-aws-ec2-instance/).
- Run
   ```
  git clone https://github.com/KevKibe/Plant-Disease-Image-Classification-using-Pytorch.git
  ```
- Start up [Docker](https://docs.docker.com) and run
  ```
  docker build -t dockerfile .
  ```
- run
  ```
  docker run -e PORT=8080 dockerfile
  ```
- You can now get predictions from
  ```
  http://<ec2-public-IP>:8080/predict
  ```
