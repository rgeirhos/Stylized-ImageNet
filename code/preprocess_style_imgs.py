#!/usr/bin/env python
"""
Functionality to preprocess a directory of images: 
Try reading them, crop etc. to desired size, ...
And if corrupt (error), move them to different directory.

This serves to speed up training by doing preprocessing beforehand. 
"""

import os
import shutil
from PIL import Image

from adain import crop_square_and_downsample
import general as g


def preprocess_directory(sourcedir,
                         targetdir,
                         excludedir,
                         img_size=(g.IMG_SIZE, g.IMG_SIZE)):
    """Try reading, converting to RGB, check size.

    If corrupt, move to excludedir.
    """
 
    assert os.path.exists(sourcedir), "sourcedir not found: "+sourcedir

    if not os.path.exists(targetdir):
        os.makedirs(targetdir)
    if not os.path.exists(excludedir):
        os.makedirs(excludedir)

    num_imgs = len(os.listdir(sourcedir))
    num_corrupt_imgs = 0

    for i, img_name in enumerate(sorted(os.listdir(sourcedir))):
  
        src = os.path.join(sourcedir, img_name)
        is_corrupt = False
        try:
            image = Image.open(src)
            image = image.convert("RGB")            
            size_x, size_y = image.size
            if size_x < img_size[0] or size_y < img_size[1]:
                print("ratio too small")
                raise IOError("too small") # this will be caught by except
            image = crop_square_and_downsample(image, img_size)
            image.save(os.path.join(targetdir, img_name))
        except:
            is_corrupt = True
            num_corrupt_imgs += 1
            dst = os.path.join(excludedir, img_name)
            shutil.copyfile(src, dst)

        print("""Completed preprocessing img {} of {}. Excluded: {} ({})""".format(
              i+1, num_imgs, is_corrupt, img_name)) 
            
    print("=> Completed preprocessing {} imgs. Exluded {} imgs (moved to {}.".format(
          num_imgs, num_corrupt_imgs, excludedir))


def preprocess_paintings():
    """Call preprocessing functionality for paintings."""

    sourcedir = g.ADAIN_RAW_PAINTINGS_DIR
    targetdir = g.ADAIN_PREPROCESSED_PAINTINGS_DIR
    excludedir = g.ADAIN_EXCLUDED_PAINTINGS_DIR

    preprocess_directory(sourcedir=sourcedir,
                         targetdir=targetdir,
                         excludedir=excludedir,
                         img_size=(g.IMG_SIZE, g.IMG_SIZE))

if __name__ == "__main__":

    preprocess_paintings()
