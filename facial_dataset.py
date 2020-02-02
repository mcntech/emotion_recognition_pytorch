# import libraries
import torch
from torch import from_numpy
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from helper_functions import horizontal_flip, vertical_flip, crop_and_pad, change_brightness
import numpy as np
import random
import os
import copy
import pandas as pd
from PIL import Image

classes = {'neutral': 1, 'anger': 2, 'surprise': 3, 'smile': 4, 'sad': 5}


class FacialDataset(Dataset):
    """ Returns data, labels
        The shape of data is 3 x 44 x 44
        The labels are of type integer
    """

    def __init__(self, image_dir, csv_path, train_test_flag):
        """
        Args:
            image_dir (string): path to image files
            csv_path (string): path to csv file
            train_test_flag: flag to indicate if train data or test data is used

        """
        self.image_dir = image_dir
        # reading the file and converting it to numpy array
        self.train_test_flag = train_test_flag
        # read the csv file
        self.data_info = pd.read_csv(csv_path)
        self.data_info['filename'] = self.data_info['filename'].apply(
            lambda x: '/home/gp/Documents/projects/facial_emotion_detection/data/train/img/' + x)
        # print(self.data_info.head())
        # Fourth column is the labels
        self.data_info['class'] = self.data_info['class'].map(classes)
        # print(self.data_info.head())
        self.label_arr = np.asarray(self.data_info.iloc[:, 3])
        # get width
        self.width = np.asarray(self.data_info.iloc[:, 1])
        # get height
        self.height = np.asarray(self.data_info.iloc[:, 2])
        # Calculate len
        self.data_len = len(self.data_info.index)
        # print(self.data_len)
        # get the bounding boxes
        self.bounding_box = pd.DataFrame(self.data_info.iloc[:, 4:8])
        # print(self.bounding_box.head())
        print('Dataset initialized with', self.data_len, 'samples.')

    def __getitem__(self, index):
        """Args:
            index (int): index of the data
        Returns:
            Tensor: specific data on index which is converted to Tensor
        """
        # Get image
        single_image_name = self.data_info['filename'].iloc[index]
        # Get single image width
        single_image_width = self.data_info['width'].iloc[index]
        # Get single image height
        single_image_height = self.data_info['height'].iloc[index]
        # Open Image
        img_as_img = Image.open(single_image_name)
        # resize image t0 224 * 224 for model
        # img_as_img = img_as_img.resize((224, 224))
        """
        #  ---- sanity check begins ----
        img_as_img.show()
        #  ---- sanity check ends ----
        """
        # img_as_img.show()
        img_as_np = np.asarray(img_as_img)
        # vertical_flip flip with probability = 0.7
        if random.random() >= 0.7 and self.train_test_flag == 'train':
            img_as_np = vertical_flip(img_as_np)
        # horizontal flip with probability = 0.6
        if random.random() >= 0.6 and self.train_test_flag == 'train':
            img_as_np = horizontal_flip(img_as_np)
        # random crop with probability = 0.7
        if random.random() >= 0.7 and self.train_test_flag == 'train':
            img_as_np = crop_and_pad(img_as_np, 32)
        # Brightness increase with probability = 0.7
        if random.random() >= 0.7 and self.train_test_flag == 'train':
            pixel_brightness = random.randint(-20, 20)
            img_as_np = change_brightness(img_as_np, pixel_brightness)

        single_img = (copy.copy(img_as_np)).astype(np.float32)
        # normalizing images with standard values for Cifar10
        single_img /= 255
        # apply transforms- convert to tensor
        # Convert numpy array to tensor
        img_as_tensor = torch.from_numpy(single_img).float()
        # Get label(class) of the image
        single_image_label = self.label_arr[index]
        # print('single_image_label :', single_image_label)
        single_image_label_as_tensor = torch.as_tensor(single_image_label)
        # get the x_min, y_min, x_max, y_max  values
        bounding_box_as_np = self.bounding_box.values
        bounding_box_as_np = bounding_box_as_np[index, :]
        # convert to tensor
        bounding_box_as_tensor = torch.as_tensor(bounding_box_as_np)
        # get iscrowd
        iscrowd = 0
        single_image_iscrowd = torch.from_numpy(np.array(iscrowd, dtype=np.float)).float()
        # get area of image
        single_image_area = torch.as_tensor(single_image_width * single_image_height)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = bounding_box_as_tensor
        my_annotation["labels"] = single_image_label_as_tensor
        my_annotation["image_id"] = torch.from_numpy(np.array(index, dtype=np.float)).float()
        my_annotation["area"] = single_image_area
        my_annotation["iscrowd"] = single_image_iscrowd

        return img_as_tensor, my_annotation

    def __len__(self):
        return self.data_len


if __name__ == "__main__":
    # pass
    train_data_dir = '/home/gp/Documents/projects/facial_emotion_detection/data/train/img/'
    csv_path = '/home/gp/Documents/projects/facial_emotion_detection/data/train/train_labels.csv'
    train_dt = FacialDataset(train_data_dir, csv_path, 'train')
    train_dt[1]
    # train_dt[0]
    # train_dt[789]
    # """
