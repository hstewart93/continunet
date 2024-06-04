"""Pre-processing module for images."""

import numpy as np


class PreProcessor:
    """Pre-process image data for inference."""

    def __init__(self, image: object, layers: int):
        self.image = image
        self.layers = layers

        self.reshape()
        self.normalise()

    def reshape(self):
        """Reshape the image data for the network. Shape must be divisible by 2 ** n layers."""
        if not isinstance(self.image.data.shape[0] / 2 ** self.layers, int) or not isinstance(
            self.image.data.shape[1] / 2 ** self.layers, int
        ):
            self.image = self.image.data[
                : self.image.data.shape[0] // (2 ** self.layers) * (2 ** self.layers),
                : self.image.data.shape[1] // (2 ** self.layers) * (2 ** self.layers),
            ]

    def normalise(self):
        """Normalise the image data."""
        self.image = (self.image.data - np.min(self.image)) / (
            np.max(self.image) - np.min(self.image)
        )
        return self.image
