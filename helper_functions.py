import tqdm
import os
import csv
import torch
import pickle
import copy
import random
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt


MIN_FONT_SIZE = 8
MAX_FONT_SIZE = 100
MAX_MSG_SIZE = len('surprised: 100.00%')


class FontLoader:
    """Load a PIL ImageFont given the width of the bounding box
    This class provides a mapping from the width of an image bounding box to the largest ImageFont that does not exceed
    the extents of the bounding box (except for very small bounding boxes with only a few pixels).
    Note: As of now only the FreeMonoBold font is supported
    Parameters
    ----------
    min_size: int, optional
        The min font size to consider (default=8)
    max_size: int, optional
        The maximum font size to consider (default=50)
    max_msg_size: int, optional
        The maximum message size to consider (default=18; length of 'surprised: 100.00%')
    """

    def __init__(self, min_size=MIN_FONT_SIZE, max_size=MAX_FONT_SIZE, max_msg_size=MAX_MSG_SIZE):
        num_fonts = max_size - min_size + 1
        self.font_sizes = np.zeros(num_fonts, dtype=np.int32)
        self.msg_widths = np.zeros(num_fonts, dtype=np.int32)

        for i in range(num_fonts):
            font_size = min_size + i
            message = ' ' * max_msg_size
            font = self.get_font(font_size)
            message_width = font.getsize(message)[0]
            self.font_sizes[i] = font_size
            self.msg_widths[i] = message_width

        self.min_size = min_size
        self.max_size = max_size
        self.max_msg_size = max_msg_size

    @staticmethod
    def get_font(font_size):
        """A static method that returns the appropriate ImageFont given the font size
        Parameters
        ----------
        font_size: int
        Returns
        -------
        ImageFont.FreeTypeFont
        """
        return ImageFont.truetype('Pillow/Tests/fonts/FreeMonoBold.ttf', font_size)

    def __call__(self, img_width):
        """Return the largest ImageFont that does not exceed the image width using the maximum message length
        Parameters
        ----------
        img_width: int
            The width of the image bounding box in pixels
        Returns
        -------
        ImageFont.FreeFontType
        """
        nearest_width_idx, = np.where(self.msg_widths < img_width)
        if len(nearest_width_idx) > 0:
            idx = nearest_width_idx[-1]
        else:
            idx = 0
        font_size = self.font_sizes[idx]
        return self.get_font(font_size)


def crop_and_pad(img_as_np, crop_size):
    """
  pad an image and crop it to original size
   Args:
        images: numpy array of images
        crop_size(int): size of cropped image(32)
    Return:
        padded and cropped image: numpy array of padded and cropped image
    """
    # print('img_as_np_before_pad', img_as_np.shape[0], img_as_np.shape[1], img_as_np.shape[2])
    pad_size = 4
    img_as_np = np.pad(img_as_np, ((0, 0), (pad_size, pad_size),
                                   (pad_size, pad_size)), mode="symmetric")
    # print('img_as_np_after_pad', img_as_np.shape[0], img_as_np.shape[1], img_as_np.shape[2])
    y_loc, x_loc = random.randint(0, pad_size), random.randint(0, pad_size)
    cropped_img = img_as_np[::, y_loc:y_loc + crop_size, x_loc:x_loc + crop_size]
    # print('cropped_img', cropped_img.shape[0], cropped_img.shape[1], cropped_img.shape[2])
    return cropped_img


def change_brightness(image, value):
    """
    Args:
        image : numpy array of image
        value : brightness
    Return :
        image : numpy array of image with brightness added and with maximum-255 and minimum-0
    """
    # image = image.astype("int16")
    image = image + value
    image[image > 255] = 255
    image[image < 0] = 0
    return image


def horizontal_flip(image):
    """
    Args:
        image : numpy array of image
    Return :
        image : numpy array of flipped image
    """
    # horizontal
    flipped_image = np.flip(image, 1)
    return flipped_image


def vertical_flip(image):
    """
    Args:
        image : numpy array of image
    Return :
        image : numpy array of flipped image
    """
    # horizontal
    flipped_image = np.flip(image, 0)
    return flipped_image
