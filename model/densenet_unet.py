from abc import ABC
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, Activation, Conv2DTranspose, Concatenate
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.models import Model


class DenseNet121Unet(Model):
    def __init__(self, input_shape):
        super(DenseNet121Unet, self).__init__()

        """ Input """
        self.inputs = Input(input_shape)

        """ Pre-trained DenseNet121 Model """
        self.densenet = DenseNet121(include_top=False, weights="imagenet", input_tensor=self.inputs)

        """ Encoder """
        self.s1 = self.densenet.get_layer("input_1").output     ## 512
        self.s2 = self.densenet.get_layer("conv1/relu").output  ## 256
        self.s3 = self.densenet.get_layer("pool2_relu").output  ## 128
        self.s4 = self.densenet.get_layer("pool3_relu").output  ## 64

        """ Bridge """
        self.b1 = self.densenet.get_layer("pool4_relu").output  ## 32

        """ Decoder """
        self.d1 = self._decoder_block(self.b1, self.s4, 512)  ## 64
        self.d2 = self._decoder_block(self.d1, self.s3, 256)  ## 128
        self.d3 = self._decoder_block(self.d2, self.s2, 128)  ## 256
        self.d4 = self._decoder_block(self.d3, self.s1, 64)   ## 512

        """ Outputs """
        self.outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(self.d4)

    def call(self, inputs, **kwargs):
        return Model(inputs=self.inputs, outputs=self.outputs, name="DensetNet121-Unet")(inputs)

    def summary_model(self):
        inputs = (None, 512, 512, 3)
        outputs = self.call(inputs)
        keras.Model(inputs=inputs, outputs=outputs, name="DensetNet121-Unet").summary()

    @staticmethod
    def _conv_block(inputs, num_filters):
        x = Conv2D(num_filters, 3, padding="same")(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(num_filters, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        return x

    def _decoder_block(self, inputs, skip_features, num_filters):
        x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
        x = Concatenate()([x, skip_features])
        x = self._conv_block(x, num_filters)
        return x


if __name__ == "__main__":
    model = DenseNet121Unet(input_shape=(512, 512, 3))
    model.summary_model()
