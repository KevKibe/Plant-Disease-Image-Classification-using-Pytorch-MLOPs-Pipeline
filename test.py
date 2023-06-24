import requests

# Set the URL of the Flask API
url = 'http://localhost:5000/predict'

# Open the image file and send it as part of the POST request
with open('test\TomatoEarlyBlight6.JPG', 'rb') as f:
    response = requests.post(url, files={'file': f})

# Print the response from the server
print(response.text)
