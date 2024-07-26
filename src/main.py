# CAPTCHA Solver
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import cv2
import torch
import pytesseract

from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from models import CNN
from dataset import CaptchaDataset
from utils import load_image, load_model

# Set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

def eval(model_path='model.pth'):
    model = load_model(model_path)
    model.eval()
    test_data_dir = 'data/test'
    test_dataset = CaptchaDataset(test_data_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    # Initialize the confusion matrix
    confusion_matrix = np.zeros((10, 10))
    # Initialize the list of predicted labels
    predicted_labels = []
    # Iterate over the test dataset
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        predicted_labels.append(predicted.item())
        confusion_matrix[int(labels), predicted.item()] += 1
    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    print(f'Accuracy: {accuracy}')
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_matrix, cmap='Blues')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    confusion_matrix_df = pd.DataFrame(confusion_matrix)
    confusion_matrix_df.to_csv('confusion_matrix.csv', index=False)
    predicted_labels_df = pd.DataFrame(predicted_labels, columns=['predicted_labels'])
    predicted_labels_df.to_csv('predicted_labels.csv', index=False)
    
def train(level, model_path=None):
    if model_path == None: # train a new model
        model = CNN()
    else:
        model = load_model(model_path)
    model = model.to(device)
    
    base_img_path = f"../dataset/labelled-captcha/level_{level}"
    label_path = f"../dataset/labelled-captcha/labels_level_{level}.csv"
    images, labels = [], []
    df = pd.read_csv(label_path) # (image_name, label)
    for i in range(len(df)):
        img = cv2.imread(os.path.join(base_img_path, df.iloc[i, 0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        labels.append(str(df.iloc[i, 1]))

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((60, 160)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = CaptchaDataset(images, labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    print('after loading')

    for epoch in range(10):  
        for imgs, labels in tqdm(dataloader):
            imgs = imgs.to(device)
            outputs = model(imgs)
            labels = F.one_hot(labels, num_classes=10).float().to(device)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")    
            
    torch.save(model.state_dict(), 'trained_model.pth')
    
def solve_captcha(img_path):
    model_path = 'trained_model.pth'
    model = load_model(model_path)
    model.eval()
    img = load_image(img_path)
    img = Image.fromarray(img)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 100)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img = transform(img)
    img = img.unsqueeze(0)
    outputs = model(img)
    _, predicted = torch.max(outputs, 1)
    predicted_label = predicted.item()
    return predicted_label

def solve_captcha_with_tesseract(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    text = pytesseract.image_to_string(img)
    return text

if __name__ == '__main__':
    # Train the model
    train(1)
    exit()
    # Evaluate the model
    eval()
    # Solve a CAPTCHA image
    img_path = 'data/test/0.png'
    predicted_label = solve_captcha(img_path)
    print(f'Predicted label: {predicted_label}')
    # Solve a CAPTCHA image using Tesseract
    text = solve_captcha_with_tesseract(img_path)
    print(f'Text extracted by Tesseract: {text}')