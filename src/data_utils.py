import cv2
import os
import sys
import tensorflow as tf
from tensorflow.keras.models import load_model

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


if __name__ == '__main__':
    from data_processing import data_preprocess_test, cut_image
else:
    PACKAGE_PARENT = '...'
    SCRIPT_DIR = os.path.join(os.getcwd(), PACKAGE_PARENT)
    sys.path.append(SCRIPT_DIR)
    from src.data_processing import data_preprocess_test, cut_image


def decode_img(img, img_width, img_height, num_channels):
    '''
        Read and convert images
    Args:
        img: Tensorflow.image -- Input image
        img_width: int -- Image width
        img_height: int -- Image height
        num_channels: int -- Number of channels on the images
    Returns:
        img: Tensorflow.image -- Image after preprocessing
    '''
    img = tf.io.read_file(img)
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_image(img, num_channels)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    img = tf.image.resize(img, (img_height, img_width))
    #
    img = tf.expand_dims(img, 0)

    return img


def sp_noise(image, prob):
    '''    Add salt and pepper noise to image
    Args:
        image: Input image
        prob: Probability of the noise
    Returns:
        output: image with noise
    '''
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def augment_generate(input_path):
    '''
    Generates augmented images for training
    Args:
        input_path: path to training images

    Returns:
        number of training images
    '''

    files = []
    for dirpath, dirnames, filenames in os.walk(input_path):
        for path in filenames:
            file = os.path.join(dirpath, path)
            if file.endswith('JPG') or file.endswith('png'):
                files.append(file)

    for img_path in files:
        img = plt.imread(img_path)

        # Flip
        horizontal_flip_img = tf.image.flip_left_right(img)
        plt.imsave(img_path[:-4] + '_horizontal_flip_img.jpg', horizontal_flip_img.numpy())
        vertical_flip_img = tf.image.flip_up_down(img)
        plt.imsave(img_path[:-4] + '_vertical_flip_img.jpg', vertical_flip_img.numpy())

        # Aumento de contraste
        high_contrast_img = tf.image.adjust_contrast(img, 2)
        plt.imsave(img_path[:-4] + '_high_contrast.jpg', high_contrast_img.numpy())

        # Diminuição de brilho
        #low_brightness_img = tf.image.adjust_brightness(img, -0.2)
        #plt.imsave(img_path[:-4] + '_low_brightness_img.jpg', low_brightness_img.numpy())

        # Filtro passa baixa(3x3)
        blurred_img = cv2.GaussianBlur(img, (3, 3), 0)
        plt.imsave(img_path[:-4] + '_blurred_img.jpg', blurred_img)

        # Inclusão de 5% de ruído
        noisy_image = sp_noise(img, 0.05)
        plt.imsave(img_path[:-4] + '_noisy_image.jpg', noisy_image)


        # rotação de 90
        img_rot90 = tf.image.rot90(img, 1)
        plt.imsave(img_path[:-4] + '_rot90.jpg', img_rot90.numpy())
        # rotação de 180
        img_rot180 = tf.image.rot90(img, 2)
        plt.imsave(img_path[:-4] + '_rot180.jpg', img_rot180.numpy())
        # rotação de 270
        img_rot270 = tf.image.rot90(img, 3)
        plt.imsave(img_path[:-4] + '_rot270.jpg', img_rot270.numpy())

    return len(files)


def model_predict(model_path, img, img_width, img_height, num_channels, class_names):
    ''' Classifies a single image

    Args:
        model_path:
        img:
        img_width:
        img_height:
        num_channels:
        class_names:

    Returns:

    '''
    # load model
    model = load_model(model_path)
    # preprocess path of image
    img = decode_img(img, img_width, img_height, num_channels)
    # predict one image
    prediction = model.predict(img)
    #
    pred_class = class_names[np.argmax(prediction, 1)[0]]

    return pred_class


def confusion_matrix_gerenate(y_true, y_pred, class_names):
    ''' Generates a confusion matrix

    Args:
        y_true: list of true labels
        y_pred: list of predicted labels
        class_names: list of classe names

    Returns:

    '''

    conf_matrix = confusion_matrix(y_true, y_pred, labels=class_names)
    #cm_norm = conf_matrix / conf_matrix.astype(np.float).sum(axis=1)
    sns.heatmap(conf_matrix, cbar=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names, annot=True)
    plt.ylabel('Classe verdadeira')
    plt.xlabel('Classe prevista')
    plt.savefig('confusion_matrix.png')

    return conf_matrix


def model_evaluate(model_path, input_path, img_height, img_width, num_channels):
    '''

    Args:
        model_path:
        input_path:
        img_height:
        img_width:
        num_channels:

    Returns:

    '''
    # Class names based on the directory structure
    class_file = 'classes.txt'
    with open(class_file, 'r') as file:
        class_names = file.read().split('\n')

    # load model
    model = load_model(model_path)
    test_data = data_preprocess_test(input_path, img_height, img_width, num_channels, class_names)
    # Accuracy and loss
    test_scores = model.evaluate(test_data)

    y_true = tf.argmax(tf.concat([y for x, y in test_data], axis=0), axis=1)
    y_pred = tf.argmax(model.predict(test_data), axis=1)

    # Using labels instead of index
    class_names = np.array(class_names)
    y_true = class_names[y_true.numpy()]
    y_pred = class_names[y_pred.numpy()]

    conf_matrix = confusion_matrix_gerenate(y_true, y_pred, class_names)
    class_report = classification_report(y_true, y_pred, output_dict=True)
    pd.DataFrame(class_report).transpose().to_csv('classification_report.csv')
    class_report = classification_report(y_true, y_pred, output_dict=False)

    return test_scores, conf_matrix, class_report

