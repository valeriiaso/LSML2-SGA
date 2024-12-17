import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from flask import Flask, make_response, request


CLASSES = os.listdir('final_project/train/simpsons_dataset')
N_CLASSES = len(CLASSES)
RESCALE_SIZE = 224
DEVICE = torch.device("cuda") if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else torch.device("cpu")

def load_model(path_to_model, device=DEVICE):
    model = models.resnet50(weights='DEFAULT')
    for params in model.parameters():
        params.requires_grad = False
    for params in model.layer4.parameters():
        params.requires_grad = True
    for params in model.fc.parameters():
        params.requires_grad = True
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, N_CLASSES)
    model = model.to(device)
    model.load_state_dict(torch.load(path_to_model, map_location=device))
    model.eval()

    return model


def load_image(path_to_image):
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    raw_image = Image.open(path_to_image)
    image = np.array(raw_image.resize((RESCALE_SIZE, RESCALE_SIZE)))
    image = np.array(image / 255, dtype='float32')
    image = transform(image)
    image = image.unsqueeze(0)

    return image


def predict_one_sample(model, file, device=DEVICE):
    image = load_image(file)
    with torch.no_grad():
        image = image.to(device)
        model.eval()
        y_hat = model(image).cpu()
        y_pred = torch.argmax(y_hat, dim=1)
    return CLASSES[y_pred]


model = load_model('final_project/model_ft.pt')

app = Flask(__name__)


@app.route('/')
def upload_image():
    return """
        <html>
            <body>
                <h1>Welcome to Springfield</h1>
                </br>
                </br>
                <p> Upload an image of a Simpson character you want to classify
                <form action="/predict" method="post" enctype="multipart/form-data">
                    <input type="file" name="data_file" class="btn btn-block"/>
                    </br>
                    </br>
                    <button type="Upload" class="btn btn-primary btn-block btn-large">Classify</button>
                </form>
            </body>
        </html>
    """

@app.route('/predict', methods=["POST"])
def predict_character():
    file = request.files['data_file']
    if not file:
        return "Image is not uploaded"
    prediction = predict_one_sample(model, file)
    response = make_response(prediction)
    return response

if __name__ == '__main__':
    app.run()