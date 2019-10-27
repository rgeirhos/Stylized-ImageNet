#!/usr/env/python

"""
General definitions and paths
"""

import argparse
import os
from os.path import join as pjoin

###########################################################
#    SETTINGS THAT NEED TO BE CHANGED BY USER
###########################################################

# TODO Dear user, please change these paths:
IMAGENET_PATH = "../data/" # TODO specify ImageNet path
STYLIZED_IMAGENET_PATH = "../datanew/" # TODO specify target path: where should Stylized-ImageNet be stored?


###########################################################
#    SETTINGS THAT USUALLY DON'T NEED TO BE CHANGED
###########################################################

IMG_SIZE = 224
ADAIN_RAW_PAINTINGS_DIR = "./paintings_raw/"
ADAIN_EXCLUDED_PAINTINGS_DIR = "./paintings_excluded/"
ADAIN_PREPROCESSED_PAINTINGS_DIR = "./paintings_preprocessed/"

ADAIN_MODEL_DIR = "./models/"
ADAIN_VGG_PATH = pjoin(ADAIN_MODEL_DIR, "vgg_normalised.pth")
ADAIN_DECODER_PATH = pjoin(ADAIN_MODEL_DIR, "decoder.pth")


assert os.path.exists(ADAIN_VGG_PATH)
assert os.path.exists(ADAIN_DECODER_PATH)
assert os.path.exists(IMAGENET_PATH)
assert os.path.exists(pjoin(IMAGENET_PATH, "train/"))
assert os.path.exists(pjoin(IMAGENET_PATH, "val/"))


def get_default_adain_args():

    # parse arguments
    parser = argparse.ArgumentParser()

    # Basic options
    parser.add_argument('--gpu', type=int, default=-1)

    parser.add_argument('--vgg', type=str, default=ADAIN_VGG_PATH)
    parser.add_argument('--decoder', type=str, default=ADAIN_DECODER_PATH)

    # Additional options
    parser.add_argument('--content_size', type=int, default=IMG_SIZE,
                        help='New (minimum) size for the content image, \
                        keeping the original size if set to 0')
    parser.add_argument('--style_size', type=int, default=IMG_SIZE,
                        help='New (minimum) size for the style image, \
                        keeping the original size if set to 0')
    parser.add_argument('--crop', action='store_true',
                        help='do center crop to create squared image')
    parser.add_argument('--save_ext', default='.jpg',
                        help='The extension name of the output image')

    # Advanced options
    parser.add_argument('--preserve_color', action='store_true',
                        help='If specified, preserve color of the content image')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='The weight that controls the degree of \
                              stylization. Should be between 0 and 1')
    parser.add_argument(
        '--style_interpolation_weights', type=str, default='',
        help='The weight for blending the style of multiple style images')

    args = parser.parse_args(args=[])

    return args
