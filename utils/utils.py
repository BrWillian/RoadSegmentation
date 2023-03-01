import os
import numpy as np
from natsort import natsorted
import cv2

COLOR_MAP = [
    (0, 0, 0),
    (0, 255, 64),
    (0, 64, 255),
    (255, 64, 0),
]

CLASSES = [
    "NONE"
    "ASPHALT",
    "POTHOLE",
    "CRACK"
]

ALL_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')


class Utils:
    @staticmethod
    def get_imgs_from_directory(directory_path: str, sort: bool = False) -> list:
        """
        Return a list of image filenames in a directory and its subdirectories.

        Args:
            directory_path (str): Path to the directory containing the images.
            sort (bool, optional): If True, sort the filenames in natural order. Defaults to False.

        Returns:
            list: List of image filenames.
        """
        image_filenames = []
        for root, _, files in os.walk(directory_path):
            for filename in files:
                if filename.lower().endswith(ALL_IMAGE_EXTENSIONS):
                    image_filenames.append(os.path.join(root, filename))

        if sort:
            image_filenames = natsorted(image_filenames)

        return image_filenames

    @staticmethod
    def write_final_mask_from_directory(path: str, ext: str = ".png") -> None:
        """
        Generate a final mask by merging three different types of binary masks (mask_lane, mask_poth, mask_crack)
        for each subdirectory in the given path.

        Args:
            path (str): The directory path containing subdirectories with binary masks.
            ext (str, optional): The file extension of the binary masks. Defaults to ".png".

        Returns:
            None
        """
        for file_name in os.listdir(path):
            sub_dir_path = os.path.join(path, file_name)
            if os.path.isdir(sub_dir_path):
                masks = []
                mask_lane = None
                mask_poth = None
                mask_crack = None
                for image_name in sorted(os.listdir(sub_dir_path)):
                    if image_name.endswith(ext):
                        img = cv2.imread(os.path.join(sub_dir_path, image_name), cv2.IMREAD_GRAYSCALE)
                        if "POTHOLE" in image_name.split("_")[-1]:
                            mask_poth = img
                        if "LANE" in image_name.split("_")[-1]:
                            mask_lane = img
                        if "CRACK" in image_name.split("_")[-1]:
                            mask_crack = img

                masks = [mask_lane, mask_poth, mask_crack]
                new_mask = Utils.merge_binary_masks(masks)

                cv2.imwrite(os.path.join(sub_dir_path, f"{file_name}_FINAL.png"), new_mask)

    @staticmethod
    def rgb_to_onehot(rgb_image: np.ndarray) -> np.array:
        """
        Faz one-hot encoding de uma imagem RGB com base em uma lista de cores.

        Args:
            image (numpy.ndarray): Array numpy contendo a imagem RGB.
            color_map (list): Lista de tuplas contendo as cores de cada classe.

        Returns:
            Um array numpy contendo a imagem one-hot encoded.
            :param rgb_image:
        """
        # Cria um array numpy com as mesmas dimensões da imagem
        height, width, _ = rgb_image.shape
        num_classes = len(COLOR_MAP)
        encoded_image = np.zeros((height, width, num_classes), dtype=np.uint8)

        # Faz one-hot encoding de cada pixel
        for i, color in enumerate(COLOR_MAP):
            indices = (rgb_image == color).all(axis=2)
            encoded_image[indices, i] = 1

        return encoded_image

    @staticmethod
    def onehot_to_rgb(onehot_image: np.ndarray) -> np.array:
        """
        Converte uma imagem one-hot encoded de volta para uma imagem RGB com cores correspondentes às classes.

        Args:
            onehot_image (numpy.ndarray): Array numpy contendo a imagem one-hot encoded.
            color_map (list): Lista de tuplas contendo as cores de cada classe.

        Returns:
            Um array numpy contendo a imagem RGB com cores correspondentes às classes.
        """
        # Calcula o índice da classe para cada pixel
        indices = np.argmax(onehot_image, axis=2)

        # Cria um array numpy com as mesmas dimensões da imagem
        height, width = indices.shape
        decoded_image = np.zeros((height, width, 3), dtype=np.uint8)

        # Preenche cada pixel com a cor correspondente à sua classe
        for i, color in enumerate(COLOR_MAP):
            indices_i = np.where(indices == i)
            decoded_image[indices_i] = color

        return decoded_image

    @staticmethod
    def merge_binary_masks(mask_list: list) -> np.ndarray:
        """
        Junta uma lista de imagens de máscaras binárias em uma só, mapeando as classes para cores específicas.

        Args:
            mask_list (list): Lista contendo as máscaras binárias de cada classe.

        Returns:
            Um array numpy contendo a imagem mesclada com as máscaras.
        """
        # Cria uma matriz vazia para a imagem mesclada
        height, width = mask_list[0].shape[:2]
        merged = np.zeros((height, width, 3), dtype=np.uint8)

        # Aplica as cores às máscaras na imagem colorida, misturando as cores
        for i, img in enumerate(mask_list):
            merged[img != 0] = COLOR_MAP[i + 1]

        # Retorna a imagem resultante
        return merged

    @staticmethod
    def merge_masks(image: np.ndarray, mask: np.ndarray):
        """
        Mescla a imagem original com a máscara, mapeando as classes para cores específicas.

        Args:
            image (numpy.ndarray): Array numpy contendo a imagem original.
            mask (numpy.ndarray): Array numpy contendo a máscara one-hot encoded.

        Returns:
            Um array numpy contendo a imagem mesclada com a máscara.
        """
        alpha = 0.5  # Define a transparência da máscara
        merged = cv2.addWeighted(image, alpha, mask, alpha, 0)

        return merged
