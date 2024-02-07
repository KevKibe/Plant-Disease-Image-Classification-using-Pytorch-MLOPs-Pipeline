import os
import io
import requests
from PIL import Image
import torchvision.transforms as transforms

base_dir = os.path.dirname(os.path.abspath(__file__))

url = 'http://192.168.1.64:3000/predict'

# image_path = "test\TomatoEarlyBlight1.JPG"
image_path = os.path.join(base_dir, '../test_data/AppleCedarRust1.JPG')

pil_image = Image.open(image_path)

transform = transforms.ToTensor()
tensor_image = transform(pil_image)

img_bytes = io.BytesIO()
transforms.ToPILImage()(tensor_image).save(img_bytes, format='PNG')

r = requests.post(url, files={'file': ('image.png', img_bytes.getvalue(), 'image/png')})

print(r.text)
