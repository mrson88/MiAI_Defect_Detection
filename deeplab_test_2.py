import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds

# Configuration
BATCH_SIZE = 16
BUFFER_SIZE = 1000
EPOCHS = 50
IMAGE_SIZE = (512, 512)
NUM_CLASSES = 21  # Adjust based on your dataset

# Load and preprocess data
def preprocess(data):
    input_image = tf.image.resize(data['image'], IMAGE_SIZE)
    input_mask = tf.image.resize(data['segmentation_mask'], IMAGE_SIZE)
    
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask = tf.cast(input_mask, tf.int32)
    
    return input_image, input_mask

dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
train_dataset = dataset['train'].map(preprocess).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
test_dataset = dataset['test'].map(preprocess).batch(BATCH_SIZE)

# Define the model
base_model = tf.keras.applications.ResNet101(input_shape=[*IMAGE_SIZE, 3], include_top=False, weights='imagenet')

# Use the activations of these layers
layer_names = [
    'conv4_block23_out',   # 16x16
    'conv5_block3_out',    # 8x8
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
down_stack.trainable = False

# Define the decoder (upsampling path)
up_stack = [
    layers.UpSampling2D(size=(2,2)),
    layers.Conv2D(256, 3, activation='relu', padding='same'),
    layers.UpSampling2D(size=(2,2)),
    layers.Conv2D(128, 3, activation='relu', padding='same'),
    layers.UpSampling2D(size=(2,2)),
    layers.Conv2D(64, 3, activation='relu', padding='same'),
]

def unet_model(output_channels):
    inputs = layers.Input(shape=[*IMAGE_SIZE, 3])
    
    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])
    
    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])
    
    # This is the last layer of the model
    last = layers.Conv2DTranspose(output_channels, 3, strides=2, padding='same')
    x = last(x)
    
    return tf.keras.Model(inputs=inputs, outputs=x)

# Create and compile the model
model = unet_model(NUM_CLASSES)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
TRAIN_LENGTH = info.splits['train'].num_examples
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

model.fit(train_dataset, 
          epochs=EPOCHS,
          steps_per_epoch=STEPS_PER_EPOCH,
          validation_data=test_dataset)

# Save the model
model.save('deeplab_model.h5')