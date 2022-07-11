# Keras
import os
import time
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
# Keras
from tensorflow.keras.layers import Input, Conv2D, Dense, BatchNormalization, Dropout, Activation, MaxPooling2D, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import concatenate

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.inception_v3 import InceptionV3


import itertools

import coremltools
from src.data_io import split_images








def create_vgg19(img_height, img_width, num_channels, num_classes, model_name='topografia'):
    """

    Args:
        img_height: int
            Input images height
        img_width: int
            Input images width
        num_channels: int
            Number of input images channels
        num_classes: int
            Number of classes/outputs
        model_name: str
            Model name

    """

    base_model = VGG19(input_shape=(img_height, img_width, num_channels), include_top=False, weights= None)
    base_model.trainable = False

    upper_layer = base_model.output
    flatten = Flatten()(upper_layer)

    dense_layer1 = Dense(4096, activation='relu')(flatten)
    dropout1 = Dropout(0.5)(dense_layer1)

    dense_layer2 = Dense(4096, activation='relu')(dropout1)
    dropout2 = Dropout(0.5)(dense_layer2)

    output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(dropout2)

    model = Model(inputs=base_model.input, outputs=output_layer, name='VGG19_' + model_name)
    model.save('vgg19_' + model_name + '.h5')


def create_inceptionv3(img_height, img_width, num_channels, num_classes, model_name='oct'):
    """

    Args:
        img_height: int
            Input images height
        img_width: int
            Input images width
        num_channels: int
            Number of input images channels
        num_classes: int
            Number of classes/outputs
        model_name: str
            Model name

    """

    base_model = InceptionV3(input_shape=(img_height, img_width, num_channels), include_top=False)
    base_model.trainable = False

    upper_layer = base_model.output

    # Avg pooling
    global_avg_pooling = GlobalAveragePooling2D()(upper_layer)
    # Dropout
    dropout = Dropout(0.5)(global_avg_pooling)
    # Output layer
    output_layer = Dense(num_classes, activation='softmax')(dropout)

    model = Model(inputs=base_model.input, outputs=output_layer, name='inceptionV3_' + model_name)
    model.save('inceptionv3_' + model_name + '.h5')


def keras_callbacks(model_path):
    ''' Define the keras callbacks during the model training

        ModelCheckpoint: Saves the best model
        EarlyStopping: Stops the training if the models stop improving
        TensorBoard: Store the training information for TensorBoard visualization
    Args:
        model_path: Tensorflow.h5
            CNN model that should be trained

    Returns:
        my_callbacks: list of callbacks
            List of callbacks

    '''
    run_logdir = os.path.join('logs', time.strftime("run_%Y_%m_%d-%H_%M_%S"))
    best_model = model_path[:-3] + '{epoch:02d}-{val_loss:.2f}.h5'
    # ModelCheckpoint
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(best_model, monitor='val_loss', save_best_only=True)
    # Early Stopping
    early_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, mode='auto')
    # tensorboard --logdir=./logs --port=6006
    tensorboard_callback = tf.keras.callbacks.TensorBoard(run_logdir, update_freq=15)

    my_callbacks = [checkpoint_callback, early_callback, tensorboard_callback]

    return my_callbacks


def train_model(model_path, train_data, epochs, validation_batches, num_training_steps, num_val_steps):
    '''
        Train the CNN model
    Args:
        model_path: Tensorflow.h5 -- CNN model that should be trained
        train_data: Tensorflow.Dataset -- Batches of training images
        epochs: int -- Number of epochs during training
        validation_batches: Tensorflow.Dataset -- Batches of validation images
        num_training_steps: int -- Number of training steps per epoch
        num_val_steps: int -- Number of steps during validation
    Returns:
    '''
    model = tf.keras.models.load_model(model_path)
    model_name = os.path.basename(model_path)[:-3]
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    my_callbacks = keras_callbacks(model_path)

    model.fit(train_data,
              steps_per_epoch=num_training_steps,
              epochs=epochs,
              validation_data=validation_batches,
              validation_steps=num_val_steps,
              callbacks=my_callbacks,
              verbose=1
              )
