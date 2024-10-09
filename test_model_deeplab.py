

import json, os, torch, cv2, numpy as np, albumentations as A
from PIL import Image; from matplotlib import pyplot as plt
from glob import glob
from torch.utils.data import random_split, Dataset, DataLoader
from albumentations.pytorch import ToTensorV2


model = torch.load("/home/mrson/python_code/hoc_deeplearning/MiAI_Defect_Detection/models/aerial_best_model.pt", weights_only=False)
# model = torch.load('/home/mrson/python_code/hoc_deeplearning/MiAI_Defect_Detection/models/aerial_best_model.pt', map_location=torch.device('cuda'), weights_only=False)
# inference(test_dl, model = model, device = device)

def load_image_for_prediction(image_path, transform):
    # Read the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply the same transformations as used in training
    transformed = transform(image=image)
    image_tensor = transformed["image"]
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor

# Define the transformation (same as used in training)
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(transpose_mask=True)
])

def predict_image(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        prediction = torch.argmax(output, dim=1)
    return prediction.squeeze().cpu().numpy()

# Load the model (assuming it's already loaded as in your code)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Example usage
image_path = "/home/mrson/python_code/hoc_deeplearning/MiAI_Defect_Detection/Semantic_segmentation_dataset/Tile 5/images/image_part_001.jpg"
image_tensor = load_image_for_prediction(image_path, transform)
prediction = predict_image(model, image_tensor, device)

# Visualize the result
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.imread(image_path)[:, :, ::-1])  # Original image
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(prediction, cmap='jet')  # You might want to use a different colormap
plt.title("Prediction")
plt.axis('off')

plt.show()