import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Any

COLOR_MAP = [
    [128, 0, 0], [0, 128, 0], [128, 128, 0],
]

CLASSES = [
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
                new_mask = Utils.mapcolors(masks)
                new_mask = cv2.cvtColor(new_mask, cv2.COLOR_RGB2GRAY)

        return new_mask

    @staticmethod
    def mapcolors(x, **kwagers) -> np.array:
        combined_im = 0
        for index, image in enumerate(x):
            mask = np.stack((image,) * 3, -1)
            mask_rows, mask_cols = np.where(mask[:, :, 1] == 255)
            mask[mask_rows, mask_cols, :] = COLOR_MAP[index]
            combined_im += mask

        if kwagers.get("visualize"):
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
