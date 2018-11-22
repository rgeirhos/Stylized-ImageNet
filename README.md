# README

## Steps to follow:
1. clone https://github.com/naoto0804/pytorch-AdaIN
2. ``bash models/download_models.sh``
3. convert models as described in pytorch-AdaIN repo; but with python3 otherwise it will not work (for me):
``python torch_to_pytorch.py --model models/vgg_normalised.t7``
``python torch_to_pytorch.py --model models/decoder.t7``
4. in Makefile, execute ``start_preprocessing``

