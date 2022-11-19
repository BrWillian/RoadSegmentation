"""
@Author Willian Antunes
"""
import os
import cv2
import numpy as np


class DatasetLoader(object):
    def __init__(self, adjust_gama, brightness_range, randomShiftScaleRotate, rescale=1./255, horizontal_flip=False):
        self.adjust_gamma = adjust_gama
        self.brightness_range = brightness_range
        self.rescale = rescale
        self.horizontal_flip = horizontal_flip
        self.randomShiftScaleRotate = randomShiftScaleRotate

    def flow_from_directory(self, path, batch_size, target_size=(None, None)):
        pass


    @staticmethod
    def adjust_gamma(image, gamma=(0.2, 1.0)):
        invGamma = 1.0 / np.random.uniform(gamma[0], gamma[1])
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    @staticmethod
    def brightness_range(image, range=(0.2, 1.0)):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        value = np.random.uniform(range[0], range[1])

        lim = 255 - (value * 100)
        v[v > lim] = 255
        v[v <= lim] += int(value * 100)

        final_hsv = cv2.merge((h, s, v))

        image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

        return image

    @staticmethod
    def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                                 sat_shift_limit=(-255, 255),
                                 val_shift_limit=(-255, 255), u=0.5):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
        h = cv2.add(h, hue_shift)
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        return image

    @staticmethod
    def randomHorizontalFlip(image, mask, u=0.5):
        if np.random.random() < u:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        return image, mask

    @staticmethod
    def randomShiftScaleRotate(image, mask, shift_limit=(-0.0625, 0.0625), scale_limit=(-0.1, 0.1),
                               rotate_limit=(-45, 45), aspect_limit=(0, 0), borderMode=cv2.BORDER_CONSTANT, u=0.5):

        if np.random.random() < u:
            height, width, channel = image.shape

            angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
            scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
            aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
            sx = scale * aspect / (aspect ** 0.5)
            sy = scale / (aspect ** 0.5)
            dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
            dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

            cc = np.math.cos(angle / 180 * np.math.pi) * sx
            ss = np.math.sin(angle / 180 * np.math.pi) * sy
            rotate_matrix = np.array([[cc, -ss], [ss, cc]])

            box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
            box1 = box0 - np.array([width / 2, height / 2])
            box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

            box0 = box0.astype(np.float32)
            box1 = box1.astype(np.float32)
            mat = cv2.getPerspectiveTransform(box0, box1)
            image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                        borderValue=(
                                            0, 0,
                                            0,))
            mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                       borderValue=(
                                           0, 0,
                                           0,))

        return image, mask
