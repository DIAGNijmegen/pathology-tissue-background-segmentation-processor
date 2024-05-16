"""
This file contains a wrapper class for loading patches from multi-resolution images.
"""

from ..errors import imageerrors as dptimageerrors
from ..errors import configerrors as dptconfigerrors
from ..utils import imagefile as dptimagefile

import multiresolutionimageinterface as mir

import numpy as np
import hashlib
import os

#----------------------------------------------------------------------------------------------------

class ImageReader(object):
    """Wrapper class for multi-resolution image reading."""

    def __init__(self, image_path, spacing_tolerance=0.25, input_channels=None, cache_path=None):
        """
        Initialize the object and open the given image.

        Args:
            image_path (str): Path of the image to load.
            spacing_tolerance (float): Pixel spacing tolerance (percentage).
            input_channels (list, None): Desired channels that are extracted for each patch. All channels returned if None.
            cache_path (str, None): Directory or file cache path. The image is copied here before opening, and deleted upon closing.

        Raises:
            ImageOpenError: The specified image cannot be opened for reading.
            ImageChannelIndexError: A channel index is outside the range of available channels.
            UnknownImageDataTypeError: Unknown data type in the image.
            UnknownImageColorTypeError: Unknown color type in the image.
            InvalidPixelSpacingToleranceError: The pixel spacing tolerance is not valid.
            NonIsotropicPixelSpacingError: Non-isotropic pixel spacing.
        """

        # Initialize the base class.
        #
        super().__init__()

        # Initialize members.
        #
        self.__image = None               # Multi-resolution image object.
        self.__path = ''                  # Path of the opened image.
        self.__cache = None               # Path of the cached image.
        self.__hash = None                # Hash of the source image file.
        self.__dtype = None               # Pixel data type.
        self.__coding = None              # Pixel color coding method.
        self.__patch = None               # Patch loader function.
        self.__levels = 0                 # Number of levels in the image.
        self.__downsamplings = []         # Downsampling factors for each level.
        self.__spacings = []              # Pixel spacings for each level.
        self.__spacing_tolerance = 0.0    # Tolerance for finding a level for the given pixel spacing.
        self.__spacing_ranges = []        # Ranges of pixel spacing considering the tolerance for search in each level.
        self.__spacing_cache = {}         # Pixel spacing to image level search cache.
        self.__shapes = []                # Image shapes for each level.
        self.__input_channels = []        # Select input channels for patch.
        self.__channels_filtered = False  # Flag indicating if the read channels are filtered or all returned.
        self.__empty_value = 0            # Empty value for reading missing parts.

        # Open image.
        #
        self.__cacheimage(image_path=image_path, cache_path=cache_path)
        self.__openimage(input_channels=input_channels)
        self.__setspacings(spacing_tolerance=spacing_tolerance)
        self.__calculateranges()

    def __cacheimage(self, image_path, cache_path):
        """
        Save the source and cached image paths and copy the image to the cache.

        Args:
            image_path (str): Path of the image to load.
            cache_path (str, None): Directory or file cache path.
        """

        # Save the source image path.
        #
        self.__path = image_path

        # Save the image cache path and copy the image to the cache location.
        #
        if cache_path is None:
            # Image caching is not enabled.
            #
            self.__cache = None
        else:
            # Calculate the target file path for caching. It have to match the source file extension.
            #
            cache_target = cache_path if os.path.splitext(image_path)[1].lower() == os.path.splitext(cache_path)[1].lower() else os.path.join(cache_path, os.path.basename(image_path))

            # Avoid caching on itself.
            #
            if not os.path.isfile(cache_target) or not os.path.samefile(image_path, cache_target):
                # Copy the source image to the cache location.
                #
                dptimagefile.copy_image(source_path=image_path, target_path=cache_target, overwrite=False)

                # Save the cached image path.
                #
                self.__cache = cache_target
            else:

                self.__cache = None

    def __file(self):
        """
        Get the path of the opened file: the original image file or the cached file.

        Returns:
            str: Path of the opened file.
        """

        return self.__cache if self.__cache else self.__path

    def __openimage(self, input_channels):
        """
        Open multi-resolution image for reading.

        Args:
            input_channels (list, None): Desired channels that are extracted for each patch.

        Raises:
            ImageOpenError: The specified image cannot be opened for reading.
            ImageChannelIndexError: A channel index is outside the range of available channels.
            UnknownImageDataTypeError: Unknown data type in the image.
            UnknownImageColorTypeError: Unknown color type in the image.
        """

        # Check if the image has been successfully opened and the required level is valid and the channels are available.
        #
        mr_image = mir.MultiResolutionImageReader().open(self.__file())

        if mr_image is None:
            raise dptimageerrors.ImageOpenError(self.__file(), 'read')

        # Disable caching.
        #
        mr_image.setCacheSize(0)

        # Check the channels.
        #
        total_channels = mr_image.getSamplesPerPixel()

        if input_channels is not None:
            if any(channel < 0 or total_channels <= channel for channel in input_channels):
                raise dptimageerrors.ImageChannelIndexError(self.__file(), input_channels, total_channels)

        # Store the opened image object.
        #
        self.__image = mr_image

        # Configure the data type and the appropriate loader function for it.
        #
        image_dtype = mr_image.getDataType()
        if image_dtype == mir.DataType_UChar:
            self.__dtype = np.uint8
            self.__patch = mr_image.getUCharPatch
        elif image_dtype == mir.UInt16:
            self.__dtype = np.uint16
            self.__patch = mr_image.getUInt16Patch
        elif image_dtype == mir.UInt32:
            self.__dtype = np.uint32
            self.__patch = mr_image.getUInt32Patch
        elif image_dtype == mir.DataType_Float:
            self.__dtype = np.float32
            self.__patch = mr_image.getFloatPatch
        else:
            raise dptimageerrors.UnknownImageDataTypeError(self.__file(), image_dtype)

        # Configure the pixel color coding method.
        #
        image_color_type = mr_image.getColorType()
        if image_color_type == mir.ColorType_Monochrome:
            self.__coding = 'monochrome'
        elif image_color_type == mir.ColorType_RGB:
            self.__coding = 'rgb'
        elif image_color_type == mir.ARGB:
            self.__coding = 'argb'
        elif image_color_type == mir.ColorType_Indexed:
            self.__coding = 'indexed'
        else:
            raise dptimageerrors.UnknownImageColorTypeError(self.__file(), image_color_type)

        # Configure the used channel indices.
        #
        available_channels = list(range(total_channels))
        self.__input_channels = list(input_channels) if input_channels is not None else available_channels
        self.__channels_filtered = self.__input_channels != available_channels

        # Configure the level count and cache the downsampling factors and the shapes of each level.
        #
        self.__levels = mr_image.getNumberOfLevels()
        self.__downsamplings = [mr_image.getLevelDownsample(level) for level in range(mr_image.getNumberOfLevels())]
        self.__shapes = [tuple(reversed(mr_image.getLevelDimensions(level))) for level in range(mr_image.getNumberOfLevels())]

    def __setspacings(self, spacing_tolerance):
        """
        Set the pixel spacing.

        Args:
            spacing_tolerance (float): Pixel spacing tolerance (percentage).

        Raises:
            InvalidPixelSpacingToleranceError: The pixel spacing tolerance is not valid.
            NonIsotropicPixelSpacingError: Non-isotropic pixel spacing.
        """

        # The tolerance must be non-negative.
        #
        if spacing_tolerance < 0.0:
            raise dptconfigerrors.InvalidPixelSpacingToleranceError(spacing_tolerance)

        # Attempt to get spacings using ASAP
        _spacings = self.__image.getSpacing()
        # Use Tifffile backup method for spacing in case ASAP fails
        if len(_spacings) != 2:
            print("Could not get spacing using ASAP, attempt retrieval of spacing using tifffile...")
            from tifffile import TiffFile
            with TiffFile(self.__file()) as tif:
                xres = tif.pages[0].tags["XResolution"].value
                yres = tif.pages[0].tags["YResolution"].value
            xres = 1.0 / ((xres[0] / xres[1]) / 10000.)
            yres = 1.0 / ((yres[0] / yres[1]) / 10000.)
            _spacings = (xres, yres)

        # Check if the spacing is isotropic. The ImageReader assumes isotropic pixel spacing. The limit of difference is 1 nanometre = 1*10^-3 micrometer.
        #
        if _spacings:
            if 0.001 < abs(_spacings[0] - _spacings[1]):
                raise dptimageerrors.NonIsotropicPixelSpacingError(self.__file(), _spacings)

        # Get the spacings and compute the ranges.
        #
        self.__spacings = [_spacings[0] * downsampling for downsampling in self.__downsamplings] if _spacings else [None for _ in range(len(self.__downsamplings))]
        print("SPACINGS: ", self.__spacings)
        self.__spacing_tolerance = spacing_tolerance

    def __calculateranges(self):
        """Calculate the acceptable spacing ranges for each level based on the pixel spacing tolerance."""

        # Calculate the ranges.
        #
        self.__spacing_ranges = [(spacing * (1.0 - self.__spacing_tolerance), spacing * (1.0 + self.__spacing_tolerance)) if spacing is not None else (None, None) for spacing in self.__spacings]

        # Since the ranges has changed, the cached spacing-level pairs are not valid any more.
        #
        self.__spacing_cache = {}

    def __rightbottompad(self, patch, height, width):
        """
        Pad the patch on the right and bottom side with the empty value.

        Args:
            patch (np.ndarray): Patch to pad.

        Returns:
            np.ndarray: Padded patch.
        """

        # Pad the patch to the target patch shape.
        #
        return np.pad(array=patch, pad_width=((0, 0), (0, height - patch.shape[1]), (0, width - patch.shape[2])), mode='constant', constant_values=(self.__empty_value,))

    @property
    def path(self):
        """
        Get the path of the opened image.

        Returns:
            str: Path of the opened image.
        """

        return self.__path

    @property
    def cache(self):
        """
        Get the path of the cached image.

        Returns:
            str, None: Path of the cached image.
        """

        return self.__cache

    @property
    def dtype(self):
        """
        Get the pixel data type.

        Returns:
            type: Pixel type.
        """

        return self.__dtype

    @property
    def coding(self):
        """
        Get the color coding. Possible values are: 'monochrome', 'rgb', 'argb', and 'indexed'.

        Returns:
            str: Color coding identifier.
        """

        return self.__coding

    @property
    def channels(self):
        """
        Get the number of channels read from the image.

        Returns:
            int: Number eof channels read from the image.
        """

        return len(self.__input_channels)

    @property
    def levels(self):
        """
        Get the number of levels in the image.

        Returns:
            int: Number of levels in the image.
        """

        return self.__levels

    @property
    def downsamplings(self):
        """
        Get the downsampling factors for each level.

        Returns:
            list: Downsampling factors for each level.
        """

        return self.__downsamplings

    @property
    def spacings(self):
        """
        Get the pixel spacing for each level.

        Returns:
            list: Pixel spacing for each level.
        """

        return self.__spacings

    @property
    def shapes(self):
        """
        Get the image shapes for each level.

        Returns:
            list: Image shapes for each level.
        """

        return self.__shapes

    @property
    def image(self):
        """
        Get the image object for low level access.

        Returns:
            mir.MultiResolutionImage: Image object.
        """

        return self.__image

    def hash(self):
        """
        Get the hash of the opened image.

        Returns:
            str: Hash of the opened image.
        """

        # Calculate the hash if necessary.
        #
        if self.__hash is None:
            with open(file=self.__file(), mode='rb') as image_file:
                self.__hash = hashlib.sha256(image_file.read()).hexdigest()

        return self.__hash

    def correct(self, spacing, level):
        """
        Correct the spacing for the given level. All the other levels will be re-calculated based on the given value and level. It is only recommended to use if the pixel spacing
        information is missing from the mask file due to image file saving error.

        Args:
            spacing (float): Pixel spacing to set (micrometer).
            level (int): Level to set the spacing.

        Raises:
            InvalidPixelSpacingError: Invalid pixel spacing value.
            ImageLevelConfigurationError: Invalid image level for the image.
        """

        # Check the pixel spacing and the level values.
        #
        if spacing < 0.0:
            raise dptimageerrors.InvalidPixelSpacingError(self.__file(), spacing)

        if level < 0 or self.__levels <= 0:
            raise dptimageerrors.ImageLevelConfigurationError(self.__file(), level)

        # Calculate the spacings.
        #
        self.__spacings = [spacing * level_downsampling / self.__downsamplings[level] for level_downsampling in self.__downsamplings]

        # Re-calculate the acceptance ranges.
        #
        self.__calculateranges()

    def level(self, spacing):
        """
        Get the level for the given pixel spacing.

        Args:
            spacing (float): Pixel spacing (micrometer).

        Returns:
            int: Best matching level of the given pixel spacing.

        Raises:
            PixelSpacingLevelError: There is no level found for the given pixel spacing and tolerance.
        """

        # Try to find the level for the spacing in the cache.
        #
        target_level = self.__spacing_cache.get(spacing, -1)

        if target_level < 0:
            # Find a level with a pixel spacing within tolerance.
            #
            #print("SPACING: ", spacing, " LEVELS: ", self.__spacing_ranges)
            for level in range(len(self.__spacing_ranges)):
                if self.__spacing_ranges[level][0] <= spacing <= self.__spacing_ranges[level][1]:
                    self.__spacing_cache[spacing] = level
                    return level

            # Cannot find a level for the pixel spacing with the given tolerance.
            #
            raise dptimageerrors.PixelSpacingLevelError(self.__file(), spacing, self.__spacing_tolerance)

        else:
            # Return the cached target level.
            #
            return target_level

    def test(self, spacing):
        """
        Test if there is a level for the given pixel spacing.

        Args:
            spacing (float): Pixel spacing (micrometer).

        Returns:
            bool: True if there is a level for the given pixel spacing.
        """

        # Try to find the level for the spacing in the cache.
        #
        target_level = self.__spacing_cache.get(spacing, -1)

        if target_level < 0:
            # Find a level with a pixel spacing within tolerance.
            #
            for level in range(len(self.__spacing_ranges)):
                if self.__spacing_ranges[level][0] <= spacing <= self.__spacing_ranges[level][1]:
                    self.__spacing_cache[spacing] = level
                    return True

            # Cannot find a level for the pixel spacing with the given tolerance.
            #
            return False

        else:
            # Return the cached target level.
            #
            return True

    def refine(self, spacing):
        """
        Get the pixel spacing of an existing level for the given pixel spacing within tolerance.

        Args:
            spacing (float): Pixel spacing (micrometer).

        Returns:
            float: Best matching pixel spacing of the closest level of the given pixel spacing.

        Raises:
            PixelSpacingLevelError: There is no level found for the given pixel spacing and tolerance.
        """

        return self.__spacings[self.level(spacing=spacing)]

    def content(self, spacing):
        """
        Load a the content of the complete image from the given pixel spacing.

        Args:
            spacing (float): Pixel spacing to use to find the target level (micrometer).

        Returns:
            (np.ndarray): The loaded image.

        Raises:
            PixelSpacingLevelError: There is no level found for the given pixel spacing and tolerance.
            ImageAlreadyClosedError: The image is already closed.
        """

        # Check if the image is still open.
        #
        if self.__image is None:
            raise dptimageerrors.ImageAlreadyClosedError(self.__file())

        # Find the appropriate level for the pixel spacing.
        #
        level = self.level(spacing=spacing)

        # Extract image.
        #
        patch = self.__patch(int(0), int(0), int(self.__shapes[level][1]), int(self.__shapes[level][0]), int(level))
        patch = patch.transpose(2, 0, 1)

        # Return the selected channels.
        #
        return patch[self.__input_channels, :, :] if self.__channels_filtered else patch

    def read(self, spacing, row, col, height, width):
        """
        Load a patch from the opened image from the best matching level for the given pixel spacing. The indices are interpreted at the given level. (Not on level 0.)

        Args:
            spacing (float): Pixel spacing to use to find the target level (micrometer).
            row (int): Row index of upper left pixel.
            col (int): Col index of upper left pixel.
            height (int): Height of patch.
            width (int): Width of patch.

        Returns:
            (np.ndarray): The loaded patch.

        Raises:
            PixelSpacingLevelError: There is no level found for the given pixel spacing and tolerance.
            ImageAlreadyClosedError: The image is already closed.
        """

        # Check if the image is still open.
        #
        if self.__image is None:
            raise dptimageerrors.ImageAlreadyClosedError(self.__file())

        # Find the appropriate level for the pixel spacing.
        #
        level = self.level(spacing=spacing)

        # Extract image patch.
        #
        patch = self.__patch(int(col * self.__downsamplings[level]),
                             int(row * self.__downsamplings[level]),
                             int(min(width,  self.__shapes[level][1] - col)),
                             int(min(height, self.__shapes[level][0] - row)),
                             int(level))

        patch = patch.transpose(2, 0, 1)

        # Pad the patch if necessary.
        #
        if patch.shape[1] != height or patch.shape[2] != width:
            patch = self.__rightbottompad(patch=patch, height=height, width=width)

        # Return the selected channels.
        #
        return patch[self.__input_channels, :, :] if self.__channels_filtered else patch

    def close(self, clear=True):
        """
        Close the image object and clear the cache. No further reading is possible after calling this function.

        Args:
            clear (bool): Remove the cached file.
        """

        if self.__image is not None:
            # Close the image.
            #
            del self.__image  # Explicitly calls cleanup (self.__image.close() no longer exists)
            self.__image = None

            # Clear the cache.
            #
            if clear and self.__cache is not None:
                dptimagefile.remove_image(image_path=self.__cache, ignore_errors=True)
