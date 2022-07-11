import argparse
import os
import sys

PACKAGE_PARENT = '...'
SCRIPT_DIR = os.path.join(os.getcwd(), PACKAGE_PARENT)
sys.path.append(SCRIPT_DIR)

from src.data_utils import model_evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-model', '--model_path', help='Path to the keras model', default='inceptionv3_inseption09-0.55.h5')
    ap.add_argument('-test_images', '--test_images', help='Path to test images', default='dataR/test_images')
    ap.add_argument('-height', '--img_height', help='Images height', default=224)
    ap.add_argument('-width', '--img_width', help='Images width', default=224)
    ap.add_argument('-channels', '--num_channels', help='Number of channels on the images', default=3)
    args = vars(ap.parse_args())

    test_scores, conf_matrix, class_report = model_evaluate(args['model_path'],
                                                            args['test_images'],
                                                            args['img_height'],
                                                            args['img_width'],
                                                            args['num_channels'])

    print('Accuracy: ', test_scores[1])
    print('Loss: ', test_scores[0])
    print(conf_matrix)
    print(class_report)


if __name__ == "__main__":
    main()
