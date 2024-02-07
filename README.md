## Description
This repository contains the code implementation of a ResNet CNN model used to classify images of plant leaves as healthy or diseased. The model also identifies the specific disease that a leaf is suffering from, if applicable. The repository also includes a link to the training data that was used to train the model, as well as instructions on how to use the model.


## Running it locally
- Clone the repository: `git clone https://github.com/KevKibe/KevKibe/Plant-Disease-Image-Classification-using-Pytorch`
- Navigate to the project directory: `cd Plant-Disease-Image-Classification-using-Pytorch`
- Install the dependencies: `pip install -r requirements.txt`
- Running the API: 
```
cd app
python main.py
```


## Deployment
- The project includes a Dockerfile for building a Docker image of the app, and Kubernetes configuration files for deploying the app to a Kubernetes cluster.
- To build the Docker image and run it:
```
docker build -t my-app .
docker run -p 4000:80 my-app
```
- To deploy the app to a Kubernetes cluster:
```
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
```

- To scale the number of pod replicas based on CPU utilization.
```
kubectl apply -f hpa.yaml
```
   
- To scale by adjusting the CPU and memory reservations for your pods.
```
kubectl apply -f vpa.yaml
```


