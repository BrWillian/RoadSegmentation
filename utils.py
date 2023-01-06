import os
import numpy as np


class Utils(object):

    @staticmethod
    def get_imgs_from_directory(path) -> list:
        data = []
        extensions = ('.jpg', '.jpeg', '.png')

        for root, path, files in os.walk(path):
            for filename in files:
                if filename.endswith(extensions):
                    data.append(filename)

        return data

    @staticmethod
    def get_mask_from_directory(path, type='.png') -> list:
        for file_name in os.listdir(path):
            sub_dir_path = path + '/' + file_name
            if os.path.isdir(sub_dir_path):
                print("-----------")
                for image_name in os.listdir(sub_dir_path):
                    if image_name.endswith(type):
                        print(image_name)
                print("-----------")
        return []
