import argparse
import os
import sys

PACKAGE_PARENT = '...'
SCRIPT_DIR = os.path.join(os.getcwd(), PACKAGE_PARENT)
sys.path.append(SCRIPT_DIR)


from src.data_utils import augment_generate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-train_images', '--train_images', help='Path to train images', default='data/train_images')
    args = vars(ap.parse_args())

    n_images = augment_generate(args['train_images'])

    print('Number of training images: ', n_images)
    print('Number of the new training set: ', n_images * 5)


if __name__ == "__main__":
    main()
