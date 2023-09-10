import numpy as np
import os
import tensorflow as tf
import time
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define data directory
data_dir = '~/dl4j-examples-data/dl4j-examples/flower_photos'
output_dir = './'
num_classes = 5  # Change this to the number of classes in your dataset

# Define image size and batch size
img_width, img_height = 224, 224
batch_size = 16

# Set the global random seed
tf.random.set_seed(42)

class TimingCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        print(f"Epoch {epoch + 1}/{epochs} - Time taken: {epoch_time:.2f} seconds")


# Data augmentation for training images
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.0,  # Disable shear
    zoom_range=0.0,   # Disable zoom
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

# Create the data generator for both training and validation
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    color_mode='grayscale',
    batch_size=batch_size,
    classes=None,  # Remove classes parameter
    class_mode='categorical',
    subset='training'  # Specify 'training' subset
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    color_mode='grayscale',
    batch_size=batch_size,
    classes=None,  # Remove classes parameter
    class_mode='categorical',
    subset='validation'  # Specify 'validation' subset
)

# Create and compile ResNet50 model
base_model = ResNet50(weights=None, include_top=False, input_tensor=Input(shape=(img_width, img_height, 1)))
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define early stopping and model checkpoint callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

# Create a ModelCheckpoint callback to save the model every epoch
checkpoint_callback = ModelCheckpoint(os.path.join(output_dir, 'model_epoch_{epoch:02d}.h5'), save_freq='epoch', verbose=1)

# Create the custom timing callback
timing_callback = TimingCallback()

# Train the model with early stopping and model checkpoint
epochs = 100  # You can set a large number of epochs; early stopping will prevent overfitting

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[early_stopping, checkpoint_callback, timing_callback]
)

# Save the final model
model.save(os.path.join(output_dir, 'resnet50_grayscale.h5'))

# Save class labels
class_labels = list(train_generator.class_indices.keys())
np.save(os.path.join(output_dir, 'class_labels.npy'), class_labels)

print("Model and labels saved to disk.")