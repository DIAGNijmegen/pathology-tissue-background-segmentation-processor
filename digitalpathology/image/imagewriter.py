"""
This file contains a wrapper class for writing patches to multi-resolution images.
"""

from ..errors import imageerrors as dptimageerrors
from ..utils import imagefile as dptimagefile

import multiresolutionimageinterface as mir

import numpy as np
import os

#----------------------------------------------------------------------------------------------------

class ImageWriter(object):
    """Wrapper class for multi-resolution image writing."""

    def __init__(self,
                 image_path,
                 shape,
                 spacing,
                 dtype,
                 coding,
                 indexed_channels=0,
                 compression=None,
                 interpolation=None,
                 tile_size=512,
                 jpeg_quality=None,
                 empty_value=0,
                 skip_empty=None,
                 cache_path=None):
        """
        Initialize the object and open the given image. Missing compression and interpolation methods and the skip empt flag are derived from the color coding: Monochrome images are
        compressed with 'lzw' and interpolated with 'nearest' method and all tiles written out. Otherwise the compression method is 'jpeg' and the interpolation is 'linear' and the skip
        empty flag is enabled.

        Args:
            image_path (str): Path of the image to write.
            shape (tuple): Shape of the image.
            spacing (float, tuple, None): Pixel spacing, uniform, or (row, column) tuple (micrometer).
            dtype (type): Data type. Values: np.uint8, np.uint16, np.uint32, np.float32.
            coding (str): Color coding of the pixels. Values: 'monochrome', 'rgb', 'argb', or 'indexed'.
            indexed_channels (int): Number of channels in case of 'indexed' color coding.
            compression (str, None): Data compression method in the image file. Values: 'raw', 'jpeg', oe 'lzw'.
            interpolation (str, None): Interpolation for calculating the image pyramid in the image file. Values: 'nearest', or 'linear'.
            tile_size (int): Tile size in the image file.
            jpeg_quality (int, None): JPEG quality (1-100) when using JPEG as compression method. If not set, the default 80 is used.
            empty_value (int): Value of the missing or padded tiles.
            skip_empty (bool, None): Skip writing out tiles that are filled with the empty value.
            cache_path (str, None): Directory or file cache path. The image is written here for writing, and copied to the target path upon closing.

        Raises:
            ImageOpenError: The specified image cannot be opened for writing.
            InvalidDataTypeError: Invalid data type configuration.
            InvalidColorTypeError: Invalid color coding configuration.
            InvalidIndexedChannelsError: The number of channels for indexed color coding is invalid.
            InvalidCompressionMethodError: Invalid compression method configuration.
            InvalidInterpolationMethodError: Invalid interpolation method configuration.
            InvalidTileSizeError: Invalid tile size configuration.
            InvalidJpegQualityError: Invalid JPEG quality configuration.
            InvalidImageShapeError: Invalid image shape configuration.
            InvalidPixelSpacingError: Invalid pixel spacing configuration.
        """

        # Initialize the base class.
        #
        super().__init__()

        # Initialize members.
        #
        self.__writer = None         # Multi-resolution image writer object.
        self.__path = ''             # Path of the opened image.
        self.__shape = None          # Shape of the image on the first level.
        self.__spacing = None        # Pixel spacing.
        self.__dtype = None          # Pixel data type.
        self.__coding = None         # Pixel color coding method.
        self.__channels = None       # Channel count.
        self.__compression = None    # Compression method in the image file.
        self.__interpolation = None  # Interpolation method for creating the image pyramid.
        self.__tile_size = None      # Tile size for writing the image.
        self.__tile_shape = None     # Expected shape of the written tile.
        self.__jpeg_quality = None   # JPEG quality.
        self.__empty_value = 0       # Empty value for writing empty or missing parts.
        self.__empty_tile = None     # Empty tile for writing empty part.
        self.__skip_empty = None     # Skip writing out of empty tiles.

        # Open image.
        #
        self.__configurecache(image_path=image_path, cache_path=cache_path)
        self.__openimage()
        self.__setparameters(dtype=dtype, coding=coding, indexed_channels=indexed_channels, compression=compression, interpolation=interpolation, tile_size=tile_size, jpeg_quality=jpeg_quality)
        self.__setdimensions(shape=shape, spacing=spacing)
        self.__configureemptytile(empty_value=empty_value, skip_empty=skip_empty)

    def __configurecache(self, image_path, cache_path):
        """
        Configure the image cache path.

        Args:
            image_path (str): Path of the image to load.
            cache_path (str, None): Directory or file cache path.
        """

        # Save the path of the file.
        #
        self.__path = image_path

        # Save the image cache path.
        #
        if cache_path is None:
            # Image caching is not enabled.
            #
            self.__cache = None
        else:
            # Calculate the target file path for caching. It have to match the source file extension.
            #
            cache_target = cache_path if os.path.splitext(image_path)[1].lower() == os.path.splitext(cache_path)[1].lower() else os.path.join(cache_path, os.path.basename(image_path))

            # Avoid self caching.
            #
            self.__cache = cache_target if not os.path.isfile(cache_target) or not os.path.samefile(image_path, cache_target) else None

    def __file(self):
        """
        Get the path of the opened file: the original image file or the cached file.

        Returns:
            str: Path of the opened file.
        """

        return self.__cache if self.__cache else self.__path

    def __openimage(self):
        """
        Open multi-resolution image for writing.

        Raises:
            ImageOpenError: The specified image cannot be opened for writing.
        """

        # Create image writer object.
        #
        self.__writer = mir.MultiResolutionImageWriter()

        # Create directory structure.
        #
        os.makedirs(os.path.dirname(self.__file()), exist_ok=True)

        # Open file for writing.
        #
        result_code = self.__writer.openFile(self.__file())

        if result_code != 0:
            raise dptimageerrors.ImageOpenError(self.__file(), 'write')

    def __setparameters(self, dtype, coding, indexed_channels, compression, interpolation, tile_size, jpeg_quality):
        """
        Configure the parameters of the multi-resolution image.

        Args:
            dtype (type): Data type.
            coding (str): Color coding of the pixels.
            indexed_channels (int): Number of channels in case of 'indexed' color coding.
            compression (str): Data compression method in the image file.
            interpolation (str): Interpolation for calculating the image pyramid in the image file.
            tile_size (int): Tile size in the image file.
            jpeg_quality (int): JPEG quality (1-100) when using JPEG as compression method.

        Raises:
            InvalidDataTypeError: Invalid data type configuration.
            InvalidColorTypeError: Invalid color coding configuration.
            InvalidIndexedChannelsError: The number of channels for indexed color coding is invalid.
            InvalidCompressionMethodError: Invalid compression method configuration.
            InvalidInterpolationMethodError: Invalid interpolation method configuration.
            InvalidTileSizeError: Invalid tile size configuration.
            InvalidJpegQualityError: Invalid JPEG quality configuration.
        """

        # Convert data type.
        #
        if dtype == np.uint8:
            dtype_param = mir.DataType_UChar
        elif dtype == np.uint16:
            dtype_param = mir.UInt16
        elif dtype == np.uint32:
            dtype_param = mir.UInt32
        elif dtype == np.float32:
            dtype_param = mir.DataType_Float
        else:
            raise dptimageerrors.InvalidDataTypeError(self.__file(), dtype)

        # Convert color coding.
        #
        if coding == 'monochrome':
            coding_param = mir.ColorType_Monochrome
            channel_count = 1
        elif coding == 'rgb':
            coding_param = mir.ColorType_RGB
            channel_count = 3
        elif coding == 'argb':
            coding_param = mir.ARGB
            channel_count = 4
        elif coding == 'indexed':
            if indexed_channels < 1:
                raise dptimageerrors.InvalidIndexedChannelsError(self.__file(), indexed_channels)

            coding_param = mir.ColorType_Indexed
            channel_count = indexed_channels
        else:
            raise dptimageerrors.InvalidColorTypeError(self.__file(), coding)

        # Convert compression method.
        #
        if compression is not None:
            if compression == 'raw':
                compression_param = mir.RAW
            elif compression == 'jpeg':
                compression_param = mir.JPEG
            elif compression == 'lzw':
                compression_param = mir.Compression_LZW
            else:
                raise dptimageerrors.InvalidCompressionMethodError(self.__file(), compression)

            compression_save = compression
        else:
            # Derive the compression method from the color coding.
            #
            if coding == 'monochrome':
                compression_param = mir.Compression_LZW
                compression_save = 'lzw'
            else:
                compression_param = mir.JPEG
                compression_save = 'jpeg'

        # Convert interpolation method.
        #
        if interpolation is not None:
            if interpolation == 'nearest':
                interpolation_param = mir.Interpolation_NearestNeighbor
            elif interpolation == 'linear':
                interpolation_param = mir.Interpolation_Linear
            else:
                raise dptimageerrors.InvalidInterpolationMethodError(self.__file(), interpolation)

            interpolation_save = interpolation
        else:
            # Derive the interpolation method from the color coding.
            #
            if coding == 'monochrome':
                interpolation_param = mir.Interpolation_NearestNeighbor
                interpolation_save = 'nearest'
            else:
                interpolation_param = mir.Interpolation_Linear
                interpolation_save = 'linear'

        # Check tile size.
        #
        if tile_size <= 0:
            raise dptimageerrors.InvalidTileSizeError(self.__file(), tile_size)

        # Check JPEG quality setting.
        #
        if jpeg_quality is not None:
            if jpeg_quality < 1 or 100 < jpeg_quality:
                raise dptimageerrors.InvalidJpegQualityError(self.__file(), jpeg_quality)

            jpeg_quality_save = jpeg_quality
        else:
            jpeg_quality_save = 80 if compression_save == 'jpeg' else jpeg_quality

        # Save the parameters.
        #
        self.__dtype = dtype
        self.__coding = coding
        self.__channels = channel_count
        self.__compression = compression_save
        self.__interpolation = interpolation_save
        self.__tile_size = tile_size
        self.__tile_shape = (channel_count, tile_size, tile_size)
        self.__jpeg_quality = jpeg_quality_save

        # Configure parameters.
        #
        self.__writer.setDataType(dtype_param)
        self.__writer.setColorType(coding_param)
        self.__writer.setCompression(compression_param)
        self.__writer.setInterpolation(interpolation_param)
        self.__writer.setTileSize(tile_size)

        if coding == 'indexed':
            self.__writer.setNumberOfIndexedColors(indexed_channels)

        if jpeg_quality_save is not None:
            self.__writer.setJPEGQuality(jpeg_quality_save)

    def __setdimensions(self, shape, spacing):
        """
        Set the shape and the pixel spacing of the multi-resolution image.

        Args:
            shape (tuple): Shape of the image.
            spacing (float, tuple, None): Pixel spacing (micrometer).

        Raises:
            InvalidImageShapeError: Invalid image shape configuration.
            InvalidPixelSpacingError: Invalid pixel spacing configuration.
        """

        # Check the shape.
        #
        if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
            raise dptimageerrors.InvalidImageShapeError(self.__file(), shape)

        # Check the spacing.
        #
        if spacing is not None:
            if type(spacing) in (tuple, list):
                if len(spacing) != 2 or spacing[0] <= 0.0 or spacing[1] <= 0.0:
                    raise dptimageerrors.InvalidPixelSpacingError(self.__file(), spacing)
            else:
                if spacing <= 0.0:
                    raise dptimageerrors.InvalidPixelSpacingError(self.__file(), spacing)

        # Save the parameters.
        #
        self.__shape = shape
        self.__spacing = spacing

        # Configure shape and pixel spacing.
        #
        self.__writer.writeImageInformation(shape[1], shape[0])

        if spacing is not None:
            if type(spacing) in (tuple, list):
                row_spacing = spacing[1]
                col_spacing = spacing[0]
            else:
                row_spacing = spacing
                col_spacing = spacing

            pixel_size_vec = mir.vector_double()
            pixel_size_vec.push_back(col_spacing)
            pixel_size_vec.push_back(row_spacing)
            self.__writer.setSpacing(pixel_size_vec)

    def __configureemptytile(self, empty_value, skip_empty):
        """
        Configure the empty tile for writing.

        Args:
            empty_value (int): Value of the missing or padded tiles.
            skip_empty (bool, None): Skip writing out tiles that are filled with the empty value.
        """

        # Prepare empty tile.
        #
        self.__empty_value = self.__dtype(empty_value)
        self.__empty_tile = np.full(shape=self.__tile_shape, fill_value=self.__empty_value, dtype=self.__dtype)

        # Save the skip empty flag.
        #
        if skip_empty is not None:
            skip_empty_save = skip_empty
        else:
            if self.__coding == 'monochrome':
                skip_empty_save = True
            else:
                skip_empty_save = False

        self.__skip_empty = skip_empty_save

    def __rightbottompad(self, tile):
        """
        Pad the tile on the right and bottom side with the empty value.

        Args:
            tile (np.ndarray): Tile to pad.

        Returns:
            np.ndarray: Padded tile.
        """

        # Pad the tile to the target tile shape.
        #
        return np.pad(array=tile,
                      pad_width=((0, self.__tile_shape[0] - tile.shape[0]), (0, self.__tile_shape[1] - tile.shape[1]), (0, self.__tile_shape[2] - tile.shape[2])),
                      mode='constant',
                      constant_values=(self.__empty_value,))

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
    def shape(self):
        """
        Get the shape of the image at the lowest level.

        Returns:
            tuple: Image shape.
        """

        return self.__shape

    @property
    def spacing(self):
        """
        Get the pixel spacing of the image at the lowest level.

        Returns:
            float, tuple: Pixel spacing.
        """

        return self.__spacing

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
        Get the channel count.

        Returns:
            int: Channel count.
        """

        return self.__channels

    @property
    def compression(self):
        """
        Get the image compression method. Possible values are: 'raw', 'jpeg', and 'lzw'.

        Returns:
            str: Compression method.
        """

        return self.__compression

    @property
    def interpolation(self):
        """
        Get the image interpolation method. Possible values are: 'nearest', and 'linear'.

        Returns:
            str: Interpolation method.
        """

        return self.__interpolation

    @property
    def tilesize(self):
        """
        Get the tile size.

        Returns:
            int: Tile size.
        """

        return self.__tile_size

    @property
    def tileshape(self):
        """
        Get the tile shape.

        Returns:
            int: Tile size.
        """

        return self.__tile_shape

    @property
    def quality(self):
        """
        Get the JPEG quality setting. It is only used when JPEG compression is set.

        Returns:
            int, None: JPEG quality.
        """

        return self.__jpeg_quality

    @property
    def emptyvalue(self):
        """
        Get the empty value.

        Returns:
            int: The value of empty or missing areas.
        """

        return self.__empty_value

    @property
    def skipempty(self):
        """
        Get the flag of skipping empty tiles.

        Returns:
            bool: Skip empty tiles flag.
        """

        return self.__skip_empty

    def fill(self, content):
        """
        Write out an array as image.

        Args:
            content (np.ndarray): Image content to write.

        Raises:
            ContentShapeMismatchError: The content does not match the configured image shape.
            ContentDimensionsMismatchError: The content dimensions does not match the configured image channels.
            ImageAlreadyClosedError: The image is already closed.
            InvalidTileAddressingError: The tile target coordinate is not valid.
            DataTypeMismatchError: Image - tile data type mismatch error.
            InvalidTileShapeError: Invalid tile shape.
        """

        # Check content shape.
        #
        if content.shape[-2:] != self.__shape:
            raise dptimageerrors.ContentShapeMismatchError(content.shape, self.__shape)

        if not (content.shape[0] == 1 and self.__coding == 'monochrome' or
                content.shape[0] == 3 and self.__coding == 'rgb' or
                content.shape[0] == 4 and self.__coding == 'argb'):
            raise dptimageerrors.ContentDimensionsMismatchError(content.shape, self.__coding)

        # Write out the image content tile by tile.
        #
        for row in range(0, content.shape[1], self.__tile_size):
            for col in range(0, content.shape[2], self.__tile_size):
                self.write(tile=content[:, row:row + self.__tile_size, col: col + self.__tile_size], row=row, col=col)

    def write(self, tile, row, col):
        """
        Write the next tile to the image. Tiles must be written in order by filling up the rows continuously. Empty tiles are replaced with empty value. Tiles smaller than the required size
        are padded. Tiles are expected in channels first order.

        The target position of the tile can be added with the 'row' and 'col' pixel addresses. The 'row' and 'col' values have to be either both set or both None. The given coordinates means
        the upper left corner of the tile to write. It is not recommended to switch between automatic and explicit addressing.

        Args:
            tile (np.ndarray, None): Tile to write. None or empty tiles are replaced with an empty value.
            row (int): Row index of upper left pixel.
            col (int): Column index of upper left pixel.

        Raises:
            ImageAlreadyClosedError: The image is already closed.
            InvalidTileAddressingError: The tile target coordinate is not valid.
            DataTypeMismatchError: Image - tile data type mismatch error.
            InvalidTileShapeError: Invalid tile shape.
        """

        # Check if the image is still open.
        #
        if self.__writer is None:
            raise dptimageerrors.ImageAlreadyClosedError(self.__file())

        # Check the coordinates.
        #
        if row % self.__tile_size != 0 or col % self.__tile_size != 0:
            raise dptimageerrors.InvalidTileAddressError(row, col)

        # Check data type.
        #
        if tile is not None and tile.dtype != self.__dtype:
            raise dptimageerrors.DataTypeMismatchError(self.__file(), self.__dtype, tile.dtype)

        # Check the input tile.
        #
        if tile is None or tile.size == 0:
            # Write out an empty (tile filled with the empty value) tile if the passed tile is None or empty sized.
            #
            if not self.__skip_empty:
                self.__writer.writeBaseImagePartToLocation(self.__empty_tile.flatten(), col, row)
        elif tile.ndim == len(self.__tile_shape):
            # Check if the tile is filled with the empty value.
            #
            if not self.__skip_empty or not np.array_equal(tile, self.__empty_tile):
                # Write out the given tile with padding if necessary.
                #
                if tile.shape != self.__tile_shape:
                    self.__writer.writeBaseImagePartToLocation(self.__rightbottompad(tile=tile).transpose((1, 2, 0)).flatten(), col, row)
                else:
                    self.__writer.writeBaseImagePartToLocation(tile.transpose((1, 2, 0)).flatten(), col, row)
        else:
            # The dimension count does not match.
            #
            raise dptimageerrors.InvalidTileShapeError(self.__file(), tile.shape)

    def close(self, clear=True):
        """
        Close the image object. No further writing is possible after calling this function.

        Args:
            clear (bool): Remove the cached file.
        """

        if self.__writer is not None:
            self.__writer.finishImage()
            self.__writer = None

            # Copy or move the result from the cache path to the target.
            #
            if self.__cache and self.__path != self.__cache:
                if clear:
                    dptimagefile.move_image(source_path=self.__cache, target_path=self.__path, overwrite=True)
                else:
                    dptimagefile.copy_image(source_path=self.__cache, target_path=self.__path, overwrite=True)
