import torch
import torch.nn.functional as F
from PIL import Image

def smart_padding(image, divisor=16):
    """ Pad the image so that its dimensions are divisible by the divisor. """
    h, w = image.shape[-2:]
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    padding = (left, right, top, bottom)
    padded_image = F.pad(image, padding, mode='constant', value=1.0)
    return padded_image, padding

def remove_padding(image, padding):
    """ Remove the padding from the image. """
    left, right, top, bottom = padding
    if right == 0:
        w_end = image.shape[-1]
    else:
        w_end = -right
    if bottom == 0:
        h_end = image.shape[-2]
    else:
        h_end = -bottom
    return image[..., top:h_end, left:w_end]