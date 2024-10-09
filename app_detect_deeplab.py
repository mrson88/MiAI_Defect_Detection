import sys
import os
import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QTextEdit, QScrollArea
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from scipy import ndimage

class ImageSegmentationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeepLabV3 Image Segmentation")
        self.setGeometry(100, 100, 1600, 900)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self.image_layout = QHBoxLayout()
        self.original_image_label = QLabel(self)
        self.segmented_image_label = QLabel(self)

        # Create scroll areas for both images
        self.original_scroll = QScrollArea()
        self.original_scroll.setWidget(self.original_image_label)
        self.original_scroll.setWidgetResizable(True)
        self.segmented_scroll = QScrollArea()
        self.segmented_scroll.setWidget(self.segmented_image_label)
        self.segmented_scroll.setWidgetResizable(True)

        self.image_layout.addWidget(self.original_scroll)
        self.image_layout.addWidget(self.segmented_scroll)
        self.layout.addLayout(self.image_layout)

        self.button_layout = QHBoxLayout()
        self.load_button = QPushButton("Load Image", self)
        self.load_button.clicked.connect(self.load_image)
        self.segment_button = QPushButton("Segment Image", self)
        self.segment_button.clicked.connect(self.segment_image)
        self.button_layout.addWidget(self.load_button)
        self.button_layout.addWidget(self.segment_button)
        self.layout.addLayout(self.button_layout)

        self.area_text = QTextEdit(self)
        self.area_text.setReadOnly(True)
        self.area_text.setMaximumHeight(100)
        self.layout.addWidget(self.area_text)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        self.transform = self.get_transform()

        self.current_image = None

    def load_model(self):
        model = torch.load("/home/mrson/python_code/hoc_deeplearning/MiAI_Defect_Detection/models/aerial_best_model.pt", map_location=self.device)
        model = model.to(self.device)
        model.eval()
        return model

    def get_transform(self):
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.bmp)")
        if file_name:
            self.current_image = cv2.imread(file_name)
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            self.display_image(self.current_image, self.original_image_label)
            print(f"Image loaded: {file_name}")

    def segment_image(self):
        if self.current_image is None:
            print("No image loaded")
            return

        image_tensor = self.load_image_for_prediction(self.current_image)
        prediction = self.predict_image(image_tensor)
        
        # Calculate areas
        areas = self.calculate_areas(prediction)
        self.display_areas(areas)
        
        # Normalize prediction to 0-1 range
        prediction_normalized = (prediction - prediction.min()) / (prediction.max() - prediction.min())
        
        # Apply colormap
        colormap = plt.get_cmap('jet')
        colored_prediction = (colormap(prediction_normalized) * 255).astype(np.uint8)
        
        # Resize the colored prediction to match the original image size
        original_height, original_width = self.current_image.shape[:2]
        colored_prediction_resized = cv2.resize(colored_prediction, (original_width, original_height))
        
        # Blend the original image with the colored prediction
        alpha = 0.5
        blended_image = cv2.addWeighted(self.current_image, 1 - alpha, colored_prediction_resized[:, :, :3], alpha, 0)
        
        # Add area labels to the image
        labeled_image = self.add_area_labels(blended_image, prediction, areas)
        
        self.display_image(labeled_image, self.segmented_image_label)
        print("Segmentation completed and displayed")

    def load_image_for_prediction(self, image):
        # Resize the image to 256x256 as required by the model
        resized_image = cv2.resize(image, (256, 256))
        transformed = self.transform(image=resized_image)
        image_tensor = transformed["image"]
        image_tensor = image_tensor.unsqueeze(0)
        return image_tensor

    def predict_image(self, image_tensor):
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            output = self.model(image_tensor)
            prediction = torch.argmax(output, dim=1)
        return prediction.squeeze().cpu().numpy()

    def calculate_areas(self, prediction):
        unique, counts = np.unique(prediction, return_counts=True)
        total_pixels = prediction.size
        return {class_id: (count, count/total_pixels*100) for class_id, count in zip(unique, counts)}

    def display_areas(self, areas):
        area_text = "Segmentation Areas:\n"
        for class_id, (pixel_count, percentage) in areas.items():
            area_text += f"Class {class_id}: {pixel_count} pixels ({percentage:.2f}%)\n"
        self.area_text.setText(area_text)

    def add_area_labels(self, image, prediction, areas):
        labeled_image = image.copy()
        for class_id, (_, percentage) in areas.items():
            mask = (prediction == class_id).astype(np.uint8)
            if mask.sum() > 0:
                # Find the centroid of the region
                cy, cx = ndimage.center_of_mass(mask)
                cy, cx = int(cy), int(cx)
                
                # Add text to the image
                text = f"{percentage:.1f}%"
                cv2.putText(labeled_image, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        return labeled_image

    def display_image(self, image, label):
        h, w, ch = image.shape
        bytes_per_line = ch * w
        qt_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        label.setPixmap(pixmap)
        label.resize(pixmap.size())
        print(f"Image displayed with size: {w}x{h}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageSegmentationApp()
    window.show()
    sys.exit(app.exec_())