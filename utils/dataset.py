"""
@Author Willian Antunes
"""

from typing import Tuple
import numpy as np
from numpy import ndarray
import cv2


class DatasetLoader:
    def __init__(self, adjust_gamma: bool, brightness_range: bool, randomShiftScaleRotate: bool,
                 rescale: float = 1. / 255,
                 horizontal_flip: bool = False) -> None:
        """
        Initializes a DatasetLoader object.

        Arguments:
            adjust_gamma (bool): The gamma adjustment should be applied.
            brightness_range (bool): whether to apply brightness adjustment.
            randomShiftScaleRotate (bool): whether to apply random shift, scale and rotation.
            rescale (float): scale factor to apply to images.
            horizontal_flip (bool): horizontal flip must be applied.
        """
        self.adjust_gamma = adjust_gamma
        self.brightness_range = brightness_range
        self.rescale = rescale
        self.horizontal_flip = horizontal_flip
        self.randomShiftScaleRotate = randomShiftScaleRotate

    def flow_from_directory(self, path: str, batch_size: int, target_size: Tuple[int, int] = (None, None)):
        """
        Generate batches of augmented data from the images in a directory.

        Args:
            path (str): The directory path containing the images.
            batch_size (int): The number of samples per batch.
            target_size (tuple): The size to resize the images to.

        Returns:
            A generator that yields batches of augmented data.
        """
        pass

    @staticmethod
    def adjust_gamma(image: np.ndarray, gamma: Tuple[float, float] = (0.2, 1.0)) -> ndarray:
        """
        Randomly adjust the gamma of an image.

        Args:
            image (np.ndarray): The input image.
            gamma (tuple): The range of gamma values to randomly sample from.

        Returns:
            The input image with gamma adjusted.
        """
        invGamma = 1.0 / np.random.uniform(gamma[0], gamma[1])
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    @staticmethod
    def brightness_range(image: np.ndarray, range: Tuple[float, float] = (0.2, 1.0)) -> ndarray:
        """
        Randomly adjust the brightness of an image.

        Args:
            image (np.ndarray): The input image.
            range (tuple): The range of brightness values to randomly sample from.

        Returns:
            The input image with brightness adjusted.
        """
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
    def randomHueSaturationValue(image: np.ndarray, hue_shift_limit: Tuple[int, int] = (-180, 180),
                                 sat_shift_limit: Tuple[int, int] = (-255, 255),
                                 val_shift_limit: Tuple[int, int] = (-255, 255), u: float = 0.5) -> ndarray:
        """
        Randomly adjust the hue, saturation, and value of an image.

        Args:
            image (np.ndarray): The input image.
            hue_shift_limit (tuple): The range of hue shift values to randomly sample from.
            sat_shift_limit (tuple): The range of saturation shift values to randomly sample from.
            val_shift_limit (tuple): The range of value shift values to randomly sample from.
            u (float): The probability of applying the transformation.

        Returns:
            The input image with hue, saturation, and value adjusted.
        """
        if np.random.random() < u:
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
    def randomHorizontalFlip(image: np.ndarray, mask: np.ndarray, u: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Randomly flips an image and its corresponding mask horizontally.

        Args:
            image (numpy.ndarray): The input image.
            mask (numpy.ndarray): The corresponding mask image.
            u (float): The probability of performing a horizontal flip. Default value is 0.5.

        Returns:
            tuple: A tuple containing the flipped image and mask.
        """
        if np.random.random() < u:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        return image, mask

    @staticmethod
    def randomShiftScaleRotate(image: np.ndarray, mask: np.ndarray, shift_limit: Tuple[float, float] = (-0.0625, 0.0625),
                               scale_limit: Tuple[float, float] = (-0.1, 0.1), rotate_limit: Tuple[int, int] = (-45, 45),
                               aspect_limit: Tuple[int, int] = (0, 0), borderMode: int = cv2.BORDER_CONSTANT,
                               u: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Randomly applies shift, scale and rotation transformations to the image and its mask with a given probability.

        Args:
            image: The input image.
            mask: The corresponding mask for the input image.
            shift_limit: The limits of the range from which the shift values are sampled.
            scale_limit: The limits of the range from which the scale values are sampled.
            rotate_limit: The limits of the range from which the rotation values are sampled.
            aspect_limit: The limits of the range from which the aspect values are sampled.
            borderMode: The border mode used when applying the transformations.
            u: The probability of applying the augmentation.

        Returns:
            The transformed image and mask.
        """
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