from utils.utils import Utils
from utils.dataset import DatasetLoader


def main():
    #images = Utils.get_imgs_from_directory("./dataset/CRACK500/traindata/")

    #Utils.write_final_mask_from_directory("/media/willian/Servidor/tcc/v1/")

    train_datagen = DatasetLoader(
        adjust_gamma=True,
        brightness_range=True,
        randomShiftScaleRotate=True,
        rescale=1./255.0,
        horizontal_flip=True
    )

    train_generator = train_datagen.flow_from_directory("/media/willian/Servidor/tcc/CPR-TRATADO/", batch_size=16,
                                             target_size=(512, 512))


    exit(0)
    # print(images)
    # loader = DatasetLoader(
    #
    # )
    # for root, path, files in os.walk("./dataset/CRACK500/traindata/"):
    #     for file in files:
    #         if file.endswith(".jpg"):
    #             abs_path = os.path.join(root, file)
    #             img = cv2.imread(abs_path)
    #             img = cv2.resize(img, (500, 500))
    #             cv2.imshow("img", img)
    #             cv2.waitKey(0)
    #             img = loader.randomHueSaturationValue(img,
    #                                                   hue_shift_limit=(
    #                                                   np.random.uniform(-25, -100), np.random.uniform(25, 100)),
    #                                                   sat_shift_limit=(
    #                                                   np.random.uniform(-45, 0), np.random.uniform(0, 45)),
    #                                                   val_shift_limit=(
    #                                                   np.random.uniform(-50, 0), np.random.uniform(0, 50))
    #                                                   )
    #             img = loader.adjust_gamma(img)
    #             cv2.imshow("img", img)
    #             cv2.waitKey(0)


if __name__ == "__main__":
    main()