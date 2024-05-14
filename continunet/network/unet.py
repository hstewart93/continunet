"""UNet model for image segmentation in keras."""

from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers import Input, BatchNormalization, Activation, Dropout
from keras.layers import Concatenate
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from tensorflow.keras.optimizers import Adam


class UNet:
    """UNet model for image segmentation."""

    def __init__(self, input_shape, n_filters=16, dropout=0.1, batchnorm=True):
        self.input_shape = input_shape
        self.n_filters = n_filters
        self.dropout = dropout
        self.batchnorm = batchnorm

    def convolutional_block(self, input_tensor, n_filters, kernel_size=3):
        """Convolutional block for UNet."""
        x = Conv2D(
            filters=n_filters,
            kernel_size=(kernel_size, kernel_size),
            kernel_initializer="he_normal",
            padding="same",
        )(input_tensor)
        if self.batchnorm:
            x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(
            filters=n_filters,
            kernel_size=(kernel_size, kernel_size),
            kernel_initializer="he_normal",
            padding="same",
        )(x)
        if self.batchnorm:
            x = BatchNormalization()(x)
        x = Activation("relu")(x)

        return x

    def encoding_block(self, input_tensor, n_filters, kernel_size=3):
        """Encoding block for UNet."""
        x = self.convolutional_block(input_tensor, n_filters, kernel_size)
        p = MaxPooling2D((2, 2), padding="same")(x)
        p = Dropout(self.dropout)(p)
        return x, p

    def decoding_block(self, input_tensor, concat_tensor, n_filters, kernel_size=3):
        """Decoding block for UNet."""
        u = Conv2DTranspose(n_filters, (3, 3), strides=(2, 2), padding="same")(input_tensor)
        u = Concatenate()([u, concat_tensor])
        u = Dropout(self.dropout)(u)
        c = self.convolutional_block(u, n_filters, kernel_size)
        return c

    def build_model(self):
        input_img = Input(self.input_shape, name="img")
        # Encoder Path
        c1, p1 = self.encoding_block(input_img, self.n_filters * 1)
        c2, p2 = self.encoding_block(p1, self.n_filters * 2)
        c3, p3 = self.encoding_block(p2, self.n_filters * 4)
        c4, p4 = self.encoding_block(p3, self.n_filters * 8)
        c5 = self.convolutional_block(p4, n_filters=self.n_filters * 16)

        # Decoder Path
        c6 = self.decoding_block(c5, c4, self.n_filters * 8)
        c7 = self.decoding_block(c6, c3, self.n_filters * 4)
        c8 = self.decoding_block(c7, c2, self.n_filters * 2)
        c9 = self.decoding_block(c8, c1, self.n_filters * 1)

        outputs = Conv2D(1, (1, 1), activation="sigmoid")(c9)
        model = Model(inputs=[input_img], outputs=[outputs])
        return model

    def compile_model(self):
        """Compile the UNet model."""
        Input(self.input_shape, name="img")
        model = self.build_model()
        model.compile(
            optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy", "iou_score"]
        )
        return model
