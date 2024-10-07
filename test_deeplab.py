import json
import os
import numpy as np
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.layers import Conv2D, UpSampling2D, Concatenate
from keras.models import Model
import cv2

# Hàm để đọc và xử lý nhiều file JSON annotation
def load_annotations_from_directory(json_dir):
    annotations = []
    image_paths = []
    
    for json_file in os.listdir(json_dir):
        if json_file.endswith('.json'):
            with open(os.path.join(json_dir, json_file), 'r') as f:
                data = json.load(f)
            
            for item in data['annotations']:
                segmentation = item['segmentation']
                image_id = item['image_id']
                image_info = next(img for img in data['images'] if img['id'] == image_id)
                
                annotation = np.zeros((image_info['height'], image_info['width']))
                for seg in segmentation:
                    poly = np.array(seg).reshape((-1, 2)).astype(np.int32)
                    cv2.fillPoly(annotation, [poly], 1)
                
                annotations.append(annotation)
                image_paths.append(os.path.join(data['image_dir'], image_info['file_name']))
    
    return np.array(annotations), image_paths

# Hàm để tạo mô hình DeepLab (giữ nguyên từ phiên bản trước)
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
    x = UpSampling2D(size=(4, 4))(x)
    
    # Low-level features
    low_level_features = base_model.get_layer('conv2_block3_out').output
    low_level_features = Conv2D(48, 1, padding='same', activation='relu')(low_level_features)
    
    x = Concatenate()([x, low_level_features])
    x = Conv2D(256, 3, padding='same', activation='relu')(x)
    x = Conv2D(256, 3, padding='same', activation='relu')(x)
    x = UpSampling2D(size=(4, 4))(x)
    
    output = Conv2D(num_classes, 1, padding='same', activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    return model

# Hàm để tải và tiền xử lý ảnh
def load_and_preprocess_image(image_path, target_size=(256, 256)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize to [0, 1]
    return img

# Generator để tải dữ liệu theo batch
def data_generator(image_paths, annotations, batch_size=8, target_size=(256, 256)):
    num_samples = len(image_paths)
    while True:
        indices = np.random.permutation(num_samples)
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]
            
            batch_images = np.array([load_and_preprocess_image(image_paths[i], target_size) for i in batch_indices])
            batch_annotations = np.array([cv2.resize(annotations[i], target_size) for i in batch_indices])
            batch_annotations = np.expand_dims(batch_annotations, axis=-1)
            
            yield batch_images, batch_annotations

# Chuẩn bị dữ liệu
json_dir = 'path/to/your/json/directory'
annotations, image_paths = load_annotations_from_directory(json_dir)

# Tạo và biên dịch mô hình
input_shape = (256, 256, 3)  # Điều chỉnh kích thước theo nhu cầu
num_classes = 2  # Số lượng lớp (background + foreground)
model = create_deeplabv3_plus(input_shape, num_classes)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Tạo generators cho training và validation
train_generator = data_generator(image_paths[:int(0.8*len(image_paths))], 
                                 annotations[:int(0.8*len(annotations))], 
                                 batch_size=8)
val_generator = data_generator(image_paths[int(0.8*len(image_paths))],
                            annotations[int(0.8*len(annotations))], 
                               batch_size=8)

# Huấn luyện mô hình
steps_per_epoch = len(image_paths) // 8
validation_steps = len(image_paths) // 8 // 5  # 20% for validation

model.fit(train_generator,
          steps_per_epoch=steps_per_epoch,
          epochs=50,
          validation_data=val_generator,
          validation_steps=validation_steps)

# Lưu mô hình
model.save('deeplab_model_multi_json.h5')