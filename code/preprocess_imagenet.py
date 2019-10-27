import sys
import argparse
import os
import shutil
import time
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.utils import save_image, make_grid
from tensorboardX import SummaryWriter
from PIL import Image

import general as g
import adain

#####################################################################
# purpuse of this file:
# preprocess complete ImageNet (train + val) with AdaIN style
# transfer to speed-up later training.
#####################################################################

parser = argparse.ArgumentParser(description='Preprocess ImageNet to create Stylized-ImageNet')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')


def main():

    global args
    args = parser.parse_args()

    # Data loading code
    traindir = os.path.join(g.IMAGENET_PATH, 'train')
    valdir = os.path.join(g.IMAGENET_PATH, 'val')

    #############################################################
    #         START STYLE TRANSFER SETUP
    #############################################################

   # style_dir = g.ADAIN_PREPROCESSED_PAINTINGS_DIR
   # assert len(os.listdir(style_dir)) == 79395

    do_style_preprocessing = False
   # num_styles = len(os.listdir(style_dir))
   # print("=> Using "+str(num_styles)+" different style images.")
   # all_styles = [[] for _ in range(num_styles)]
   # for i, name in enumerate(sorted(os.listdir(style_dir))):
   #     all_styles[i] = os.path.join(style_dir, name)

    transfer_args = g.get_default_adain_args()
    transferer = adain.SmoothMe(transfer_args)
    print("=> Succesfully loaded style transfer algorithm.")

    smooth_loader = adain.SmoothLoader(smooth_transferer = transferer,
                                     rng = np.random.RandomState(seed=49809),
                                     do_preprocessing = do_style_preprocessing)

    smooth_transfer = smooth_loader.get_style_tensor_function
    print("=> Succesfully created style loader.")

    #############################################################
    #         CREATING DATA LOADERS
    #############################################################
 
    class MyDataLoader():
        """Convenient data loading class."""

        def __init__(self, root,
                     transform = transforms.ToTensor(),
                     target_transform = None,
                     batch_size = args.batch_size,
                     num_workers = args.workers,
                     shuffle = False,
                     sampler = None):

             self.dataset = datasets.ImageFolder(root = root,
                                                 transform = transform,
                                                 target_transform = target_transform)
             self.loader = torch.utils.data.DataLoader(
                               dataset = self.dataset,
                               batch_size = batch_size,
                               shuffle = shuffle,
                               sampler = sampler,
                               num_workers = num_workers,
                               pin_memory = True)


    default_transforms = transforms.Compose([
                                  transforms.Resize(256),
                                  transforms.CenterCrop(g.IMG_SIZE),
                                  transforms.ToTensor()])

    val_loader = MyDataLoader(root = valdir,
                              transform = default_transforms,
                              shuffle = False,
                              sampler = None)

    train_loader = MyDataLoader(root = traindir,
                                transform = default_transforms,
                                shuffle = False,
                                sampler = None)

    print("=> Succesfully created all data loaders.")
    print("")

    #############################################################
    #         PREPROCESS DATASETS
    #############################################################

    print("Preprocessing validation data:")
    preprocess(data_loader = val_loader,
               input_transforms = [smooth_transfer],
               sourcedir = valdir,
               targetdir = os.path.join(g.STYLIZED_IMAGENET_PATH, "val/"))

    print("Preprocessing training data:")
    preprocess(data_loader = train_loader,
               input_transforms = [smooth_transfer],
               sourcedir = traindir,
               targetdir = os.path.join(g.STYLIZED_IMAGENET_PATH, "train/"))



def preprocess(data_loader, sourcedir, targetdir,
               input_transforms = None):
    """Preprocess ImageNet with certain transformations.

    Keyword arguments:
    sourcedir -- a directory path, e.g. /bla/imagenet/train/
                 where subdirectories correspond to single classes
                 (need to be filled with .JPEG images)
    targetdir -- a directory path, e.g. /bla/imagenet-new/train/
                 where sourcedir will be mirrored, except
                 that images will be preprocessed and saved
                 as .png instead of .JPEG
    input_transforms -- a list of transformations that will
                        be applied (e.g. style transfer)
    """

    counter = 0
    current_class = None
    current_class_files = None

    # create list of all classes
    all_classes = sorted(os.listdir(sourcedir))

    for i, (input, target) in enumerate(data_loader.loader):

        # apply manipulations
        for transform in input_transforms:
            input = transform(input)

        for img_index in range(input.size()[0]):

            # for each image in a batch:
            # - determine ground truth class
            # - transform image
            # - save transformed image in new directory
            #   with the same class name

            # the mapping between old and new filenames
            # is achieved by looking at the indices of
            # the sorted(os.listdir()) results.

            source_class = all_classes[target[img_index]]
            source_classdir = os.path.join(sourcedir, source_class)
            assert os.path.exists(source_classdir)

            target_classdir = os.path.join(targetdir, source_class)
            if not os.path.exists(target_classdir):
                os.makedirs(target_classdir)

            if source_class != current_class:
                # moving on to new class:
                # start counter (=index) by 0, update list of files 
                # for this new class
                counter = 0
                current_class_files = sorted(os.listdir(source_classdir))
            
            current_class = source_class

            target_img_path = os.path.join(target_classdir, 
                                           current_class_files[counter].replace(".JPEG", ".png")) 

            save_image(tensor = input[img_index,:,:,:],
                       filename = target_img_path)
            counter += 1

        if i % args.print_freq == 0:
            print('Progress: [{0}/{1}]\t'
                   .format(
                   i, len(data_loader.loader)))


if __name__ == '__main__':
    main()
