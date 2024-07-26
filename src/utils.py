# def load_image, load_model
import cv2
from models import CNN
import torch

def load_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def load_model(model_path):
    model = CNN()
    model.load_state_dict(torch.load(model_path))
    return model