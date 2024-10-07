import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Conv2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
import cv2

def create_deeplabv3_plus(input_shape, num_classes):
    base_model = ResNet50(input_shape=input_shape, weights='imagenet', include_top=False)
    
    # ASPP (Atrous Spatial Pyramid Pooling)
    x = base_model.output
    x1 = Conv2D(256, 1, padding='same', activation='relu')(x)
    x2 = Conv2D(256, 3, padding='same', activation='relu', dilation_rate=6)(x)
    x3 = Conv2D(256, 3, padding='same', activation='relu', dilation_rate=12)(x)
    x4 = Conv2D(256, 3, padding='same', activation='relu', dilation_rate=18)(x)
    
    x = Concatenate()([x1, x2, x3, x4])
    x = Conv2D(256, 1, padding='same', activation='relu')(x)
    
    # Adjust the upsampling factor to match the low-level features
    x = UpSampling2D(size=(4, 4))(x)
    
    # Low-level features
    low_level_features = base_model.get_layer('conv2_block3_out').output
    low_level_features = Conv2D(48, 1, padding='same', activation='relu')(low_level_features)
    
    # Ensure that the dimensions match before concatenation
    x = Concatenate()([x, low_level_features])
    x = Conv2D(256, 3, padding='same', activation='relu')(x)
    x = Conv2D(256, 3, padding='same', activation='relu')(x)
    x = UpSampling2D(size=(4, 4))(x)
    
    output = Conv2D(num_classes, 1, padding='same', activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    return model

def load_and_preprocess_image(image_path, mask_path, target_size=(256, 256)):
    # Load and preprocess the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize to [0, 1]

    # Load and preprocess the mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, target_size)
    mask = np.expand_dims(mask, axis=-1)
    mask = (mask > 0).astype(np.float32)  # Binary mask

    return img, mask

def data_generator(image_dir, mask_dir, batch_size=8, target_size=(256, 256)):
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))
    
    num_samples = len(image_files)
    while True:
        indices = np.random.permutation(num_samples)
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]
            
            batch_images = []
            batch_masks = []
            for i in batch_indices:
                img, mask = load_and_preprocess_image(
                    os.path.join(image_dir, image_files[i]),
                    os.path.join(mask_dir, mask_files[i]),
                    target_size
                )
                batch_images.append(img)
                batch_masks.append(mask)
            
            yield np.array(batch_images), np.array(batch_masks)

def main():
    # Set up data directories
    image_dir = '/home/mrson/python_code/hoc_deeplearning/MiAI_Defect_Detection/NV_public_defects/Class1'
    mask_dir = '/home/mrson/python_code/hoc_deeplearning/MiAI_Defect_Detection/NV_public_defects/Class1_mask'

    # Set up model parameters
    input_shape = (256, 256, 3)  # Adjust size as needed
    num_classes = 2  # Number of classes (background + foreground)
    batch_size = 8
    epochs = 50

    # Create and compile the model
    model = create_deeplabv3_plus(input_shape, num_classes)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Create generators for training and validation
    train_generator = data_generator(image_dir, mask_dir, batch_size=batch_size)
    val_generator = data_generator(image_dir, mask_dir, batch_size=batch_size)  # You might want to use separate directories for validation

    # Calculate steps per epoch
    steps_per_epoch = len(os.listdir(image_dir)) // batch_size
    validation_steps = len(os.listdir(image_dir)) // batch_size // 5  # 20% for validation

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=validation_steps
    )

    # Save the model
    model.save('deeplab_model_image_data.h5')

    # Optional: Plot training history
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(122)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

if __name__ == "__main__":
    main()