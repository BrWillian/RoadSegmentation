import os
import cv2
import numpy as np
import segmentation_models as sm
from utils.utils import Utils
from typing import Tuple
from tqdm import tqdm

os.environ["SM_FRAMEWORK"] = "tf.keras"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

model = sm.Unet('seresnet34', classes=4, activation='softmax', encoder_weights='imagenet')

model.load_weights(filepath='weights/best_weights_with_ratio.hdf5')


def load_image(path: str) -> Tuple[np.ndarray, np.ndarray]:
    img_ori = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = Utils.resize_image(img_ori, (512, 512))
    #img = cv2.resize(img_ori, (512, 512), interpolation=cv2.INTER_AREA)
    img = img.reshape((1,) + img.shape)
    img = np.array(img, np.float32) / 255

    return img, img_ori


def main(path = "/home/willian/CPR-TRATADO/test/images/"):
    for file_name in tqdm(os.listdir(path)):
        sub_dir_path = os.path.join(path, file_name)
        img, img_ori = load_image(sub_dir_path)
        preds = model.predict_on_batch(img)
        preds = np.squeeze(preds)
        preds_decode = Utils.onehot_to_rgb(preds)
        w, h = img_ori.shape[:-1]
        mask = Utils.resize_image(img_ori, (512, 512))
        #mask = cv2.resize(preds_decode, (h, w), interpolation=cv2.INTER_AREA)
        img_merged = Utils.merge_masks(mask, preds_decode)
        cv2.imwrite(os.path.join(path, f"{file_name.strip('_RAW.jpg')}_FINAL.png"), img_merged)
        #cv2.imshow("preds", img_merged)
        #cv2.waitKey(0)


if __name__ == "__main__":
    main()
