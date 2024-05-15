"""UNet model for image segmentation in keras."""

import numpy as np

from keras.layers import (
    Activation,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    Input,
    MaxPooling2D,
)
from keras.models import Model
from tensorflow.keras.optimizers import Adam


class Unet:
    """UNet model for image segmentation."""

    def __init__(
        self,
        input_shape,
        filters=16,
        dropout=0.05,
        batch_normalisation=True,
        trained_model=None,
        image=None,
    ):
        self.input_shape = input_shape
        self.filters = filters
        self.dropout = dropout
        self.batch_normalisation = batch_normalisation
        self.trained_model = trained_model
        self.image = image

    def convolutional_block(self, input_tensor, filters, kernel_size=3):
        """Convolutional block for UNet."""
        convolutional_layer = Conv2D(
            filters=filters,
            kernel_size=(kernel_size, kernel_size),
            kernel_initializer="he_normal",
            padding="same",
        )
        batch_normalisation_layer = BatchNormalization()
        relu_layer = Activation("relu")

        if self.batch_normalisation:
            return relu_layer(batch_normalisation_layer(convolutional_layer(input_tensor)))
        return relu_layer(convolutional_layer(input_tensor))

    def encoding_block(self, input_tensor, filters, kernel_size=3):
        """Encoding block for UNet."""
        convolutional_block = self.convolutional_block(input_tensor, filters, kernel_size)
        max_pooling_layer = MaxPooling2D((2, 2), padding="same")
        dropout_layer = Dropout(self.dropout)

        return convolutional_block, dropout_layer(max_pooling_layer(convolutional_block))

    def decoding_block(self, input_tensor, concat_tensor, filters, kernel_size=3):
        """Decoding block for UNet."""
        transpose_convolutional_layer = Conv2DTranspose(
            filters, (3, 3), strides=(2, 2), padding="same"
        )
        skip_connection = Concatenate()(
            [transpose_convolutional_layer(input_tensor), concat_tensor]
        )
        dropout_layer = Dropout(self.dropout)
        return self.convolutional_block(dropout_layer(skip_connection), filters, kernel_size)

    def build_model(self):
        """Build the UNet model."""
        input_image = Input(self.input_shape, name="img")

        # Encoder Path
        first_convolutional_tensor, first_feature_map = self.encoding_block(
            input_image, self.filters * 1
        )
        second_convolutional_tensor, second_feature_map = self.encoding_block(
            first_feature_map, self.filters * 2
        )
        third_convolutional_tensor, third_feature_map = self.encoding_block(
            second_feature_map, self.filters * 4
        )
        fourth_convolutional_tensor, fourth_feature_map = self.encoding_block(
            third_feature_map, self.filters * 8
        )
        latent_convolutional_tensor = self.convolutional_block(
            fourth_feature_map, filters=self.filters * 16
        )

        # Decoder Path
        sixth_convolutional_tensor = self.decoding_block(
            latent_convolutional_tensor, fourth_convolutional_tensor, self.filters * 8
        )
        seventh_convolutional_tensor = self.decoding_block(
            sixth_convolutional_tensor, third_convolutional_tensor, self.filters * 4
        )
        eighth_convolutional_tensor = self.decoding_block(
            seventh_convolutional_tensor, second_convolutional_tensor, self.filters * 2
        )
        ninth_convolutional_tensor = self.decoding_block(
            eighth_convolutional_tensor, first_convolutional_tensor, self.filters * 1
        )

        outputs = Conv2D(1, (1, 1), activation="sigmoid")(ninth_convolutional_tensor)
        model = Model(inputs=[input_image], outputs=[outputs])
        return model

    def compile_model(self):
        """Compile the UNet model."""
        Input(self.input_shape, name="img")
        model = self.build_model()
        model.compile(
            optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy", "iou_score"]
        )
        return model

    def decode_image(self):
        """Returns images decoded by a trained model."""
        model = self.compile_model()
        if self.trained_model is None or self.image is None:
            raise ValueError("Trained model and image arguments are required to decode image.")
        if isinstance(self.image, np.ndarray) is False:
            raise TypeError("Image must be a numpy array.")
        if len(self.image.shape) != 4:
            raise ValueError("Image must be 4D numpy array for example (1, 256, 256, 1).")
        if self.image.shape[3] != 1:
            raise ValueError("Input image must be grayscale.")
        if self.image.shape[0] % 256 != 0 and self.image.shape[1] % 256 != 0:
            raise ValueError("Image shape should be divisible by 256.")

        model.load_weights(self.trained_model)
        return model.predict(self.image)
