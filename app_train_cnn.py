import argparse
import os
import sys

PACKAGE_PARENT = '...'
SCRIPT_DIR = os.path.join(os.getcwd(), PACKAGE_PARENT)
sys.path.append(SCRIPT_DIR)

from src.data_processing import data_preprocess
from src.classifiers import train_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-model', '--model_path', help='Path to the keras model', default='inceptionv3_inseption.h5')
    ap.add_argument('-i', '--input_path', help='Directory containing all images', default='dataR')
    ap.add_argument('-e', '--epochs', help='Number of epochs during training', default=100)
    ap.add_argument('-height', '--img_height', help='Images height', default=224)
    ap.add_argument('-width', '--img_width', help='Images width', default=224)
    ap.add_argument('-channels', '--num_channels', help='Number of channels on the images', default=3)
    ap.add_argument('-batch_size', '--batch_size', help='Size of batches during training', default=8)
    ap.add_argument('-val_batch_size', '--val_batch_size', help='Size of batches during validation', default=8)
    ap.add_argument('-cache', '--cache', help='Inform if the data should be cached', default=True)
    args = vars(ap.parse_args())

    train_batches, val_batches, num_training_steps, num_val_steps = data_preprocess(args['input_path'],
                                                                                    args['img_height'],
                                                                                    args['img_width'],
                                                                                    args['num_channels'],
                                                                                    args['batch_size'],
                                                                                    args['val_batch_size'],
                                                                                    args['cache'])

    train_model(args['model_path'], train_batches, args['epochs'], val_batches, num_training_steps, num_val_steps)


if __name__ == '__main__':
    main()
