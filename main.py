import cv2
import os
import numpy as np
from dataset import DatasetLoader


def main():
    loader = DatasetLoader('')
    for root, path, files in os.walk("./dataset/CRACK500/traindata/"):
        for file in files:
            if file.endswith(".jpg"):
                abs_path = os.path.join(root, file)
                img = cv2.imread(abs_path)
                img = cv2.resize(img, (500, 500))
                cv2.imshow("img", img)
                cv2.waitKey(0)
                # img = loader.randomHueSaturationValue(img,
                #                                       hue_shift_limit=(
                #                                       np.random.uniform(-25, -100), np.random.uniform(25, 100)),
                #                                       sat_shift_limit=(
                #                                       np.random.uniform(-45, 0), np.random.uniform(0, 45)),
                #                                       val_shift_limit=(
                #                                       np.random.uniform(-50, 0), np.random.uniform(0, 50))
                #                                       )
                img = loader.brightness_range(img)
                cv2.imshow("img", img)
                cv2.waitKey(0)


if __name__ == "__main__":
    main()