"""
This file contains a wrapper class for reading and writing plain images.
"""

from ..errors import imageerrors as dptimageerrors

import imageio
import os

#----------------------------------------------------------------------------------------------------

class PlainImage(object):
    """Wrapper class for plain images."""

    def __init__(self):
        """Initialize the object."""

        # Initialize the base class.
        #
        super().__init__()

        self.__path = ''       # Path of the image.
        self.__content = None  # Image content.

    def read(self, image_path):
        """
        Open an image.

        Args:
            image_path (str): Image path to open.
        """

        # Read the image file content.
        #
        self.__path = image_path
        self.__content = imageio.imread(uri=image_path)

        # Reorganize to channels first, add single channel if necessary.
        #
        if 3 == self.__content.ndim:
            self.__content = self.__content.transpose((2, 0, 1))
        else:
            self.__content = self.__content[None, ...]

    def write(self, image_path, jpeg_quality=None):
        """
        Write an image.

        Args:
            image_path (str): Path of the image to write.
            jpeg_quality (int, None): JPEG quality (1-100) when using JPEG as compression method. If not set, the default 80 is used.

        Raises:
            MissingPlainImageContentError: The content of the image is not set.
        """

        if self.__content is None:
            raise dptimageerrors.MissingPlainImageContentError(image_path)

        # Format specific settings.
        #
        if os.path.splitext(image_path)[1].lower() == '.png':
            keyword_args = {'compress_level': 9}
        elif os.path.splitext(image_path)[1].lower() in ['.jpg', '.jpeg']:
            keyword_args = {'quality ': 80 if jpeg_quality is None else jpeg_quality}
        else:
            keyword_args = {}

        # Write out the image.
        #
        imageio.imwrite(uri=image_path, im=self.__content.transpose((1, 2, 0)).squeeze(), **keyword_args)

    @property
    def path(self):
        """
        Get the path of the opened image.

        Returns:
            str: Path of the opened image.
        """

        return self.__path

    @property
    def channels(self):
        """
        Get the number of channels.

        Returns:
            int: Number of channels.
        """

        return None if self.__content is None else self.__content.shape[0]

    @property
    def shape(self):
        """
        Get the shape of the image.

        Returns:
            tuple: Image shape without the channels.
        """

        return None if self.__content is None else self.__content.shape[-2:]

    @property
    def content(self):
        """
        Content of the image in channels first order. In case of a single channel, the length of the first 'channel' dimension is 1.

        Returns:
            np.ndarray, None: Image content as numpy array.
        """

        return self.__content

    def fill(self, content):
        """
        Set the content of the image. The array must be in channels first order ir there are multiple channels.

        Args:
            content (np.ndarray): Image content.

        Raises:
            EmptyPlainImageContentError: The content array is None.
            InvalidContentShapeError: The shape of the content is invalid.
        """

        if content is None:
            raise dptimageerrors.EmptyPlainImageContentError()

        if content.ndim != 3 or content.shape[0] not in [1, 3]:
            raise dptimageerrors.InvalidContentShapeError(content.shape)

        # Save the content.
        #
        self.__content = content

    def clear(self):
        """Clear the content array and the source path."""

        # Clear the content.
        #
        self.__path = ''
        self.__content = None
