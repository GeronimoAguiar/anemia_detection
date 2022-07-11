import argparse
import os
import sys

PACKAGE_PARENT = '...'
SCRIPT_DIR = os.path.join(os.getcwd(), PACKAGE_PARENT)
sys.path.append(SCRIPT_DIR)

from src.classifiers import create_vgg19


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument('-height', '--img_height', help='Images height', default=224)
    ap.add_argument('-width', '--img_width', help='Images width', default=224)
    ap.add_argument('-channels', '--num_channels', help='Number of channels on the images', default=1)
    ap.add_argument('-classes', '--num_classes', help='Number of classes', default=2)
    ap.add_argument('-name', '--model_name', help='Model name', default='vgg19')

    args = vars(ap.parse_args())

    create_vgg19(args['img_height'], args['img_width'], args['num_channels'], args['num_classes'], args['model_name'])


if __name__ == "__main__":
    main()