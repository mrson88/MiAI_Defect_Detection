import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
os.environ["SM_FRAMEWORK"] = "tf.keras"
import tensorflow as tf
from tensorflow import keras
import segmentation_models as sm

# Set environment variable for segmentation_models


# Print TensorFlow version for debugging
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {tf.keras.__version__}")

# Set segmentation_models framework
sm.set_framework("tf.keras")
print(f"Segmentation Models framework: {sm.framework()}")

# Define variables
data_path = "/home/mrson/python_code/hoc_deeplearning/MiAI_Defect_Detection/NV_public_defects"
w, h = 512, 512
batch_size = 16

# Set backbone and get preprocessing function
BACKBONE = "resnet34"
preprocess_input = sm.get_preprocessing(BACKBONE)

# Dataset class
class Dataset:
    def __init__(self, image_path, mask_path, w, h):
        self.image_path = image_path
        self.mask_path = mask_path
        self.w = w
        self.h = h

    def __getitem__(self, i):
        image = cv2.imread(self.image_path[i])
        image = cv2.resize(image, (self.w, self.h), interpolation=cv2.INTER_AREA)
        image = preprocess_input(image)

        mask = cv2.imread(self.mask_path[i], cv2.IMREAD_UNCHANGED)
        image_mask = cv2.resize(mask, (self.w, self.h), interpolation=cv2.INTER_AREA)

        image_mask = [(image_mask == v) for v in [1]]
        image_mask = np.stack(image_mask, axis=-1).astype('float32')

        return image.astype('float32'), image_mask

# Dataloader class
class Dataloader(tf.keras.utils.Sequence):
    def __init__(self, dataset, batch_size, shape, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shape = shape
        self.indexes = np.arange(self.shape)

    def __getitem__(self, i):
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        return tuple(batch)

    def __len__(self):
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)

# Function to load image and mask paths
def load_path(data_path):
    classes = ['Class1', 'Class2', 'Class3', 'Class4', 'Class5', 'Class6']
    
    normal_image_path = []
    normal_mask_path = []
    for class_ in classes:
        current_folder = os.path.join(data_path, class_)
        for file in os.listdir(current_folder):
            if file.endswith("png") and (not file.startswith(".")):
                image_path = os.path.join(current_folder, file)
                mask_path = os.path.join(current_folder + "_mask", file)
                normal_mask_path.append(mask_path)
                normal_image_path.append(image_path)

    defect_image_path = []
    defect_mask_path = []
    for class_ in classes:
        class_ = class_ + "_def"
        current_folder = os.path.join(data_path, class_)
        for file in os.listdir(current_folder):
            if file.endswith("png") and (not file.startswith(".")):
                image_path = os.path.join(current_folder, file)
                mask_path = os.path.join(current_folder + "_mask", file)
                defect_mask_path.append(mask_path)
                defect_image_path.append(image_path)
                
    idx = random.sample(range(len(normal_mask_path)), len(defect_mask_path))

    normal_mask_path_new = [normal_mask_path[id] for id in idx]
    normal_image_path_new = [normal_image_path[id] for id in idx]

    image_path = normal_image_path_new + defect_image_path
    mask_path = normal_mask_path_new + defect_mask_path

    return image_path, mask_path

# Main execution
if __name__ == "__main__":
    # Load image and mask paths
    image_path, mask_path = load_path(data_path)

    # Split data into train and test sets
    image_train, image_test, mask_train, mask_test = train_test_split(image_path, mask_path, test_size=0.2)

    # Create datasets and dataloaders
    train_dataset = Dataset(image_train, mask_train, w, h)
    test_dataset = Dataset(image_test, mask_test, w, h)

    train_loader = Dataloader(train_dataset, batch_size, shape=len(image_train), shuffle=True)
    test_loader = Dataloader(test_dataset, batch_size, shape=len(image_test), shuffle=True)

    # Initialize model
    opt = tf.keras.optimizers.Adam(0.001)
    model = sm.Unet(BACKBONE, encoder_weights="imagenet", classes=1, activation="sigmoid", input_shape=(512, 512, 3), encoder_freeze=True)
    loss = sm.losses.categorical_focal_dice_loss
    metrics = [sm.metrics.iou_score]

    model.compile(optimizer=opt, loss=loss, metrics=metrics)

    # Train model
    is_train = True
    if is_train:
        from tensorflow.keras.callbacks import ModelCheckpoint
        filepath = "checkpoint.keras"
        callback = ModelCheckpoint(filepath, monitor='val_iou_score', verbose=1, save_best_only=True, mode='max')

        history = model.fit(
            train_loader, 
            validation_data=test_loader, 
            epochs=50, 
            callbacks=[callback]
        )

        # Plot training history
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['iou_score'], label='Training IOU')
        plt.plot(history.history['val_iou_score'], label='Validation IOU')
        plt.title('Model IOU')
        plt.xlabel('Epoch')
        plt.ylabel('IOU Score')
        plt.legend()

        plt.tight_layout()
        plt.show()

    else:
        # Load model for testing
        model.load_weights("checkpoint.keras")

        # Test on random samples
        ids = range(len(image_test))
        index = random.sample(ids, 10)

        for id in index:
            image = cv2.imread(image_test[id])
            image = cv2.resize(image, (512, 512))
            image = preprocess_input(image.astype('float32'))
            
            mask_predict = model.predict(image[np.newaxis, :, :, :])

            image_mask = cv2.imread(mask_test[id], cv2.IMREAD_UNCHANGED)
            image_mask = cv2.resize(image_mask, (512, 512))

            plt.figure(figsize=(15, 5))
            plt.subplot(131)
            plt.title("Product Image")
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            
            plt.subplot(132)
            plt.title("Actual Defect")
            plt.imshow(image_mask, cmap='gray')
            plt.axis('off')
            
            plt.subplot(133)
            plt.title("Predicted Defect")
            z = mask_predict[0, :, :, 0]
            plt.imshow(z, cmap='gray')
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()

    print("Process completed successfully!")