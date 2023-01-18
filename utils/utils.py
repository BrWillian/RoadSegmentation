import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Any

COLOR_MAP = [
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
]

CLASSES = [
    "background"
    "POTHOLE",
    "ASPHALT",
    "CRACK"
]


class Utils(object):
    @staticmethod
    def get_imgs_from_directory(path: str) -> list:
        data = []
        extensions = ('.jpg', '.jpeg', '.png')

        for root, path, files in os.walk(path):
            for filename in files:
                if filename.endswith(extensions):
                    data.append(filename)

        return data

    @staticmethod
    def get_mask_from_directory(path: str, ext: str = '.png') -> np.array:
        new_mask = None
        for file_name in os.listdir(path):
            sub_dir_path = path + file_name
            if os.path.isdir(sub_dir_path):
                masks = []
                for image_name in os.listdir(sub_dir_path):
                    if image_name.endswith(ext):
                        img = cv2.imread(sub_dir_path + "/" + image_name, cv2.IMREAD_GRAYSCALE)
                        masks.append(img)
                new_mask = Utils.map_colors(masks)

        return new_mask

    @staticmethod
    def rgb_to_onehot(rgb_image):
        colormap = {k: v for k, v in enumerate(COLOR_MAP)}
        num_classes = len(colormap)
        shape = rgb_image.shape[:2] + (num_classes,)
        encoded_image = np.zeros(shape, dtype=np.int8)
        for i, cls in enumerate(colormap):
            encoded_image[:, :, i] = np.all(rgb_image.reshape((-1, 3)) == colormap[i], axis=1).reshape(shape[:2])
        return encoded_image

    @staticmethod
    def onehot_to_rgb(onehot):
        colormap = {k: v for k, v in enumerate(COLOR_MAP)}
        single_layer = np.argmax(onehot, axis=-1)
        output = np.zeros(onehot.shape[:2] + (3,))
        for k in colormap.keys():
            output[single_layer == k] = colormap[k]
        return np.uint8(output)

    @staticmethod
    def map_colors(x, **kwargs) -> np.array:
        combined_im = 0
        for index, image in enumerate(x, start=1):
            mask = np.stack((image,) * 3, -1)
            mask_rows, mask_cols = np.where(mask[:, :, 1] == 255)
            mask[mask_rows, mask_cols, :] = COLOR_MAP[index]
            combined_im += mask

        if kwargs.get("visualize"):
            b, g, r = cv2.split(combined_im)
            fig = plt.figure(figsize=(12, 3))
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.imshow(np.stack((b,) * 3, -1))
            ax2 = fig.add_subplot(1, 3, 2)
            ax2.imshow(np.stack((g,) * 3, -1))
            ax3 = fig.add_subplot(1, 3, 3)
            ax3.imshow(np.stack((r,) * 3, -1))
            plt.show()

        return combined_im
