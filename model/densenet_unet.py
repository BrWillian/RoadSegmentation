import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model


class DenseNet121Unet(Model):
    def __init__(self, num_classes):
        super(DenseNet121Unet, self).__init__()
        pass

    def conv_block(self, inputs, num_filters: int):
        x = Conv2D(num_filters, 3, padding="same")(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(num_filters, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        return x