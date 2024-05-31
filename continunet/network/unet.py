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
        layers=4,
        output_activation="sigmoid",
    ):
        self.input_shape = input_shape
        self.filters = filters
        self.dropout = dropout
        self.batch_normalisation = batch_normalisation
        self.trained_model = trained_model
        self.image = image
        self.layers = layers
        self.output_activation = output_activation

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
        current = input_image

        # Encoder Path
        convolutional_tensors = []
        for layer in range(self.layers):
            convolutional_tensor, current = self.encoding_block(
                current, self.filters * (2 ** layer)
            )
            convolutional_tensors.append((convolutional_tensor))

        # Latent Convolutional Block
        latent_convolutional_tensor = self.convolutional_block(
            current, filters=self.filters * 2 ** self.layers
        )

        # Decoder Path
        current = latent_convolutional_tensor
        for layer in reversed(range(self.layers)):
            current = self.decoding_block(
                current, convolutional_tensors[layer], self.filters * (2 ** layer)
            )

        outputs = Conv2D(1, (1, 1), activation=self.output_activation)(current)
        model = Model(inputs=[input_image], outputs=[outputs])
        return model

    def compile_model(self):
        """Compile the UNet model."""
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
