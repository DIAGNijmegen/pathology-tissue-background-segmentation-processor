"""
Image related errors classes.
"""

from . import errorbase as dpterrorbase

#----------------------------------------------------------------------------------------------------

class DigitalPathologyImageError(dpterrorbase.DigitalPathologyError):
    """Error base class for all image errors."""

    def __init__(self, *args):
        """
        Initialize the object.

        Args:
            *args: Argument list.
        """

        # Initialize base class.
        #
        super().__init__(*args)

#----------------------------------------------------------------------------------------------------

class ImageOpenError(DigitalPathologyImageError):
    """Raise when the image file cannot be opened."""

    def __init__(self, image_path, open_mode):
        """
        Initialize the object.

        Args:
            image_path (str): Image file path.
            open_mode (str): Open mode.
        """

        # Initialize base class.
        #
        super().__init__('Cannot open image: \'{path}\' in \'{mode}\' mode.'.format(path=image_path, mode=open_mode))

        # Store custom data.
        #
        self.image_path = image_path
        self.open_mode = open_mode

#----------------------------------------------------------------------------------------------------

class InvalidPixelSpacingError(DigitalPathologyImageError):
    """Raise when the configured pixel spacing is invalid."""

    def __init__(self, image_path, pixel_spacing):
        """
        Initialize the object.

        Args:
            image_path (str): Image file path.
            pixel_spacing (float, tuple): Pixel spacing (micrometer).
        """

        # Initialize base class.
        #
        super().__init__('Invalid {spacing} pixel spacing configured for image: \'{path}\'.'.format(spacing=pixel_spacing, path=image_path))

        # Store custom data.
        #
        self.image_path = image_path
        self.pixel_spacing = pixel_spacing

#----------------------------------------------------------------------------------------------------

class PixelSpacingLevelError(DigitalPathologyImageError):
    """Raise when there is no corresponding level for the given pixel pacing (with tolerance)."""

    def __init__(self, image_path, pixel_spacing, spacing_tolerance):
        """
        Initialize the object.

        Args:
            image_path (str): Image file path.
            pixel_spacing (float): Pixel spacing (micrometer).
            spacing_tolerance (float): Pixel spacing tolerance (percentage).
        """

        # Initialize base class.
        #
        super().__init__('Level not found for {spacing} pixel spacing, with {tolerance} relative tolerance in image: \'{path}\'.'.format(spacing=pixel_spacing,
                                                                                                                                         tolerance=spacing_tolerance,
                                                                                                                                         path=image_path))

        # Store custom data.
        #
        self.image_path = image_path
        self.pixel_spacing = pixel_spacing
        self.spacing_tolerance = spacing_tolerance

#----------------------------------------------------------------------------------------------------

class ImageChannelIndexError(DigitalPathologyImageError):
    """Raise when channels are requested outside the available channels."""

    def __init__(self, image_path, input_channels, available_channels):
        """
        Initialize the object.

        Args:
            image_path (str): Image file path.
            input_channels (list): List of channels.
            available_channels (int): Number of available channels.
        """

        # Initialize base class.
        #
        super().__init__('Input channels {channels} are out of {available} range in image: \'{path}\'.'.format(channels=input_channels, available=available_channels-1, path=image_path))

        # Store custom data.
        #
        self.image_path = image_path
        self.input_channels = input_channels
        self.available_channels = available_channels

#----------------------------------------------------------------------------------------------------

class NonIsotropicPixelSpacingError(DigitalPathologyImageError):
    """Raise when pixel spacing in the image is non-isotropic."""

    def __init__(self, image_path, pixel_spacing):
        """
        Initialize the object.

        Args:
            image_path (str): Image file path.
            pixel_spacing (tuple): Pixel spacing in 2 dimensions
        """

        # Initialize base class.
        #
        super().__init__('Non isotropic {spacing} pixel spacing in image: \'{path}\'.'.format(spacing=pixel_spacing, path=image_path))

        # Store custom data.
        #
        self.image_path = image_path
        self.pixel_spacing = pixel_spacing

#----------------------------------------------------------------------------------------------------

class ImageFormatError(DigitalPathologyImageError):
    """Raise when the format of the image file is unknown."""

    def __init__(self, image_path):
        """
        Initialize the object.

        Args:
            image_path (str): Image file path.
        """

        # Initialize base class.
        #
        super().__init__('Format of the image is unknown: \'{path}\'.'.format(path=image_path))

        # Store custom data.
        #
        self.path = image_path

#----------------------------------------------------------------------------------------------------

class EmptyImageLevelListError(DigitalPathologyImageError):
    """Raise when the list of levels to use is empty."""

    def __init__(self, image_path):
        """
        Initialize the object.

        Args:
            image_path (str): Image file path.
        """

        # Initialize base class.
        #
        super().__init__('Empty image level list for image: \'{path}\'.'.format(path=image_path))

        # Store custom data.
        #
        self.path = image_path

#----------------------------------------------------------------------------------------------------

class UnknownImageDataTypeError(DigitalPathologyImageError):
    """Raise when the data type identifier in an image is unknown."""

    def __init__(self, image_path, data_type_value):
        """
        Initialize the object.

        Args:
            image_path (str): Image file path.
            data_type_value (int): Data type identifier.
        """

        # Initialize base class.
        #
        super().__init__('The {data_type} data type is not valid in image: \'{path}\'.'.format(data_type=data_type_value, path=image_path))

        # Store custom data.
        #
        self.image_path = image_path
        self.data_type_value = data_type_value

#----------------------------------------------------------------------------------------------------

class InvalidDataTypeError(DigitalPathologyImageError):
    """Raise when the data type identifier for an image is invalid."""

    def __init__(self, image_path, data_type):
        """
        Initialize the object.

        Args:
            image_path (str): Image file path.
            data_type (type): Data type identifier.
        """

        # Initialize base class.
        #
        super().__init__('Invalid \'{data_type}\' data type for image: \'{path}\'.'.format(data_type=data_type, path=image_path))

        # Store custom data.
        #
        self.image_path = image_path
        self.data_type = data_type

#----------------------------------------------------------------------------------------------------

class DataTypeMismatchError(DigitalPathologyImageError):
    """Raise when the data type identifier of the written tile does not match the configured data type of the image."""

    def __init__(self, image_path, image_data_type, tile_data_type):
        """
        Initialize the object.

        Args:
            image_path (str): Image file path.
            image_data_type (type): Tile data type identifier.
            tile_data_type (type): Tile data type identifier.
        """

        # Initialize base class.
        #
        super().__init__('The \'{tile}\' data type of the tile does not match the {image} data type of the image: \'{path}\'.'.format(tile=tile_data_type, image=image_data_type, path=image_path))

        # Store custom data.
        #
        self.image_path = image_path
        self.image_data_type = image_data_type
        self.tile_data_type = tile_data_type

#----------------------------------------------------------------------------------------------------

class UnknownImageColorTypeError(DigitalPathologyImageError):
    """Raise when the color type identifier in an image is unknown."""

    def __init__(self, image_path, color_type_value):
        """
        Initialize the object.

        Args:
            image_path (str): Image file path.
            color_type_value (int): Color type identifier.
        """

        # Initialize base class.
        #
        super().__init__('The {color_type} color type is not valid in image: \'{path}\'.'.format(color_type=color_type_value, path=image_path))

        # Store custom data.
        #
        self.image_path = image_path
        self.color_type_value = color_type_value

#----------------------------------------------------------------------------------------------------

class InvalidColorTypeError(DigitalPathologyImageError):
    """Raise when the color type identifier for an image is invalid."""

    def __init__(self, image_path, color_type):
        """
        Initialize the object.

        Args:
            image_path (str): Image file path.
            color_type (str): Color type identifier.
        """

        # Initialize base class.
        #
        super().__init__('Invalid \'{color_type}\' color type for image: \'{path}\'.'.format(color_type=color_type, path=image_path))

        # Store custom data.
        #
        self.image_path = image_path
        self.color_type = color_type

#----------------------------------------------------------------------------------------------------

class InvalidIndexedChannelsError(DigitalPathologyImageError):
    """Raise when the number of channels for indexed color coding is invalid."""

    def __init__(self, image_path, indexed_channels):
        """
        Initialize the object.

        Args:
            image_path (str): Image file path.
            indexed_channels (int): Number of channels.
        """

        # Initialize base class.
        #
        super().__init__('Invalid number of channels {channels} for indexed color type for image: \'{path}\'.'.format(channels=indexed_channels, path=image_path))

        # Store custom data.
        #
        self.image_path = image_path
        self.indexed_channels = indexed_channels

#----------------------------------------------------------------------------------------------------

class InvalidCompressionMethodError(DigitalPathologyImageError):
    """Raise when the compression method identifier for an image is invalid."""

    def __init__(self, image_path, compression_method):
        """
        Initialize the object.

        Args:
            image_path (str): Image file path.
            compression_method (str): Compression method identifier.
        """

        # Initialize base class.
        #
        super().__init__('Invalid \'{compression}\' compression method for image: \'{path}\'.'.format(compression=compression_method, path=image_path))

        # Store custom data.
        #
        self.image_path = image_path
        self.compression_method = compression_method

#----------------------------------------------------------------------------------------------------

class InvalidInterpolationMethodError(DigitalPathologyImageError):
    """Raise when the interpolation method identifier for an image is invalid."""

    def __init__(self, image_path, interpolation_method):
        """
        Initialize the object.

        Args:
            image_path (str): Image file path.
            interpolation_method (str): Interpolation method identifier.
        """

        # Initialize base class.
        #
        super().__init__('Invalid \'{interpolation}\' interpolation method for image: \'{path}\'.'.format(interpolation=interpolation_method, path=image_path))

        # Store custom data.
        #
        self.image_path = image_path
        self.interpolation_method = interpolation_method

#----------------------------------------------------------------------------------------------------

class InvalidTileSizeError(DigitalPathologyImageError):
    """Raise when the tile size for an image is invalid."""

    def __init__(self, image_path, tile_size):
        """
        Initialize the object.

        Args:
            image_path (str): Image file path.
            tile_size (int): Tile size.
        """

        # Initialize base class.
        #
        super().__init__('Invalid \'{tile_size}\' tile size for image: \'{path}\'.'.format(tile_size=tile_size, path=image_path))

        # Store custom data.
        #
        self.image_path = image_path
        self.tile_size = tile_size

#----------------------------------------------------------------------------------------------------

class InvalidJpegQualityError(DigitalPathologyImageError):
    """Raise when the JPEG quality setting for an image is invalid."""

    def __init__(self, image_path, jpeg_quality):
        """
        Initialize the object.

        Args:
            image_path (str): Image file path.
            jpeg_quality (int): JPEG quality.
        """

        # Initialize base class.
        #
        super().__init__('Invalid \'{jpeg_quality}\' JPEG quality for image: \'{path}\'.'.format(jpeg_quality=jpeg_quality, path=image_path))

        # Store custom data.
        #
        self.image_path = image_path
        self.jpeg_quality = jpeg_quality

#----------------------------------------------------------------------------------------------------

class InvalidImageShapeError(DigitalPathologyImageError):
    """Raise when the configured shape for an image is invalid."""

    def __init__(self, image_path, image_shape):
        """
        Initialize the object.

        Args:
            image_path (str): Image file path.
            image_shape (tuple): Image shape
        """

        # Initialize base class.
        #
        super().__init__('Invalid \'{shape}\' image shape for image: \'{path}\'.'.format(shape=image_shape, path=image_path))

        # Store custom data.
        #
        self.image_path = image_path
        self.image_shape = image_shape

#----------------------------------------------------------------------------------------------------

class ImageLevelConfigurationError(DigitalPathologyImageError):
    """Raise when the image file does not contain the given level."""

    def __init__(self, image_path, image_level):
        """
        Initialize the object.

        Args:
            image_path (str): Image file path.
            image_level (int, list): Image level.
        """

        # Initialize base class.
        #
        super().__init__('Level {level} is not valid for image: \'{path}\'.'.format(level=image_level, path=image_path))

        # Store custom data.
        #
        self.image_path = image_path
        self.image_level = image_level

#----------------------------------------------------------------------------------------------------

class UnfixableImageSpacingError(DigitalPathologyImageError):
    """Raise when the shape and spacing of the image file cannot be matched with the mask."""

    def __init__(self, reference_image_path, reference_image_shapes, template_image_path, template_image_shapes):
        """
        Initialize the object.

        Args:
            reference_image_path (str): Reference image file path.
            reference_image_shapes (list): Reference image shape on each level.
            template_image_path (str): Template image file path.
            template_image_shapes (tuple): Template image shape on each level.
        """

        # Initialize base class.
        #
        super().__init__('Cannot fix the missing spacings of \'{template}\' image with the \'{reference}\' reference image.'.format(template=template_image_path, reference=reference_image_path))

        # Store custom data.
        #
        self.reference_image_path = reference_image_path
        self.reference_image_shapes = reference_image_shapes
        self.template_image_path = template_image_path
        self.template_image_shapes = template_image_shapes

#----------------------------------------------------------------------------------------------------

class ImageShapeMismatchError(DigitalPathologyImageError):
    """Raise when the shape and spacing of the image file cannot be matched with the mask."""

    def __init__(self, reference_image_path, reference_image_shapes, reference_image_spacings, template_image_path, template_image_shape, template_image_spacing):
        """
        Initialize the object.

        Args:
            reference_image_path (str): Reference image file path.
            reference_image_shapes (list): Shape of the reference image on each level.
            reference_image_spacings (list): Pixel spacing of the reference image on each level.
            template_image_path (str): Template image file path.
            template_image_shape (tuple): Template image shape.
            template_image_spacing (float): Template image spacing (micrometer).
        """

        # Initialize base class.
        #
        super().__init__('Image shape mismatch of: \'{reference}\' and \'{template}\' with {spacing} spacing.'.format(reference=reference_image_path,
                                                                                                                      template=template_image_path,
                                                                                                                      spacing=template_image_spacing))

        # Store custom data.
        #
        self.reference_image_path = reference_image_path
        self.reference_image_shapes = reference_image_shapes
        self.reference_image_spacings = reference_image_spacings
        self.template_image_path = template_image_path
        self.template_image_shape = template_image_shape
        self.template_image_spacing = template_image_spacing

#----------------------------------------------------------------------------------------------------

class StatShapeMismatchError(DigitalPathologyImageError):
    """Raise when the size of the stats and mask files cannot be matched."""

    def __init__(self, stat_path, stat_shape, mask_path, mask_shape, mask_spacing):
        """
        Initialize the object.

        Args:
            stat_path (str): Stat file path.
            stat_shape (tuple): Stat shape.
            mask_path (str): Mask image file path.
            mask_shape (tuple): Mask image shape.
            mask_spacing (float): Mask image pixel spacing (micrometer).
        """

        # Initialize base class.
        #
        super().__init__('Stat - mask shape mismatch: {stat} and {mask}.'.format(stat=stat_shape, mask=mask_shape))

        # Store custom data.
        #
        self.stat_path = stat_path
        self.stat_shape = stat_shape
        self.mask_path = mask_path
        self.mask_shape = mask_shape
        self.mask_spacing = mask_spacing

#----------------------------------------------------------------------------------------------------

class StatSpacingMismatchError(DigitalPathologyImageError):
    """Raise when the spacing of the stats and mask files cannot be matched."""

    def __init__(self, stat_path, stat_spacing, mask_path, mask_spacings, spacing_tolerance):
        """
        Initialize the object.

        Args:
            stat_path (str): Stat file path.
            stat_spacing (float): Stat pixel spacing (micrometer).
            mask_path (str): Mask image file path.
            mask_spacings (list): Mask image pixel spacing of all levels.
            spacing_tolerance (float): Pixel spacing tolerance (percentage).
        """

        # Initialize base class.
        #
        super().__init__('Stat - mask spacing mismatch: {stat} and {mask} with {tolerance} relative tolerance.'.format(stat=stat_spacing, mask=mask_spacings, tolerance=spacing_tolerance))

        # Store custom data.
        #
        self.stat_path = stat_path
        self.stat_spacing = stat_spacing
        self.mask_path = mask_path
        self.mask_spacings = mask_spacings
        self.spacing_tolerance = spacing_tolerance

#----------------------------------------------------------------------------------------------------

class ContentShapeMismatchError(DigitalPathologyImageError):
    """Raise when the written out content does not match the configured image shape."""

    def __init__(self, content_shape, image_shape):
        """
        Initialize the object.

        Args:
            content_shape (tuple): Content array shape.
            image_shape (tuple): Configured image shape.
        """

        # Initialize base class.
        #
        super().__init__('The written {content} content shape is not suitable for the configured {image} image shape.'.format(content=content_shape, image=image_shape))

        # Store custom data.
        #
        self.content_shape = content_shape
        self.image_shape = image_shape

#----------------------------------------------------------------------------------------------------

class ContentDimensionsMismatchError(DigitalPathologyImageError):
    """Raise when the written out content does not match the configured image color coding."""

    def __init__(self, content_shape, color_coding):
        """
        Initialize the object.

        Args:
            content_shape (tuple): Content array shape.
            color_coding (str): Configured color coding.
        """

        # Initialize base class.
        #
        super().__init__('The written {content} content shape is not suitable for the configured \'{coding}\' color coding.'.format(content=content_shape, coding=color_coding))

        # Store custom data.
        #
        self.content_shape = content_shape
        self.color_coding = color_coding

#----------------------------------------------------------------------------------------------------

class ImageAlreadyClosedError(DigitalPathologyImageError):
    """Raise when the accessed image is already closed."""

    def __init__(self, image_path):
        """
        Initialize the object.

        Args:
            image_path (str): Image file path.
        """

        # Initialize base class.
        #
        super().__init__('The image is already closed: \'{path}\'.'.format(path=image_path))

        # Store custom data.
        #
        self.path = image_path

#----------------------------------------------------------------------------------------------------

class InvalidTileShapeError(DigitalPathologyImageError):
    """Raise when the shape of the tile to write is invalid."""

    def __init__(self, image_path, tile_shape):
        """
        Initialize the object.

        Args:
            image_path (str): Image file path.
            tile_shape (tuple): Tile shape.
        """

        # Initialize base class.
        #
        super().__init__('The {shape} tile shape is not valid for image: \'{path}\'.'.format(shape=tile_shape, path=image_path))

        # Store custom data.
        #
        self.image_path = image_path
        self.tile_shape = tile_shape

#----------------------------------------------------------------------------------------------------

class InvalidTileAddressError(DigitalPathologyImageError):
    """Raise when the tile coordinates are not multiple of the tile size."""

    def __init__(self, row, col):
        """
        Initialize the object.

        Args:
            row (int, None): Row index.
            col (int, None): Column index.
        """

        # Initialize base class.
        #
        super().__init__('Invalid tile address: ({row}, {col}).'.format(row=row, col=col))

        # Store custom data.
        #
        self.row = row
        self.col = col

#----------------------------------------------------------------------------------------------------

class AnnotationOpenError(DigitalPathologyImageError):
    """Raise when the annotation file cannot be opened."""

    def __init__(self, annotation_path):
        """
        Initialize the object.

        Args:
            annotation_path (str): Annotation file path.
        """

        # Initialize base class.
        #
        super().__init__('Cannot open annotation: \'{path}\'.'.format(path=annotation_path))

        # Store custom data.
        #
        self.annotation_path = annotation_path

#----------------------------------------------------------------------------------------------------

class InvalidFileMissingAnnotationGroupError(DigitalPathologyImageError):
    """Raise a when used annotation group is missing from the annotation file."""

    def __init__(self, annotation_path, available_groups, missing_group):
        """
        Initialize the object.

        Args:
            annotation_path (str): Annotation file path.
            available_groups (list): Available annotation groups.
            missing_group (str): Missing annotation group.
        """

        # Initialize base class.
        #
        super().__init__('The referenced \'{missing}\' group in not in the {available} list of available groups, in annotation: \'{path}\'.'.format(missing=missing_group,
                                                                                                                                                    available=available_groups,
                                                                                                                                                    path=annotation_path))

        # Store custom data.
        #
        self.annotation_path = annotation_path
        self.available_groups = available_groups
        self.missing_group = missing_group

#----------------------------------------------------------------------------------------------------

class InvalidFileUnknownAnnotationTypeError(DigitalPathologyImageError):
    """Raise when the annotation file contains annotation of an unknown type."""

    def __init__(self, annotation_path, annotation_type):
        """
        Initialize the object.

        Args:
            annotation_path (str): Annotation file path.
            annotation_type (int): Annotation type.
        """

        # Initialize base class.
        #
        super().__init__('Unknown annotation type id \'{id}\', in annotation: \'{path}\'.'.format(id=annotation_type, path=annotation_path))

        # Store custom data.
        #
        self.annotation_path = annotation_path
        self.annotation_type = annotation_type

#----------------------------------------------------------------------------------------------------

class InvalidAnnotationPathError(DigitalPathologyImageError):
    """Raise when the annotation target file path is invalid."""

    def __init__(self, annotation_path):
        """
        Initialize the object.

        Args:
            annotation_path (str): Annotation file path.
        """

        # Initialize base class.
        #
        super().__init__('Invalid annotation target file path: \'{path}\'.'.format(path=annotation_path))

        # Store custom data.
        #
        self.annotation_path = annotation_path

#----------------------------------------------------------------------------------------------------

class InvalidMaskImagePathError(DigitalPathologyImageError):
    """Raise when the mask image target file path is invalid."""

    def __init__(self, image_path):
        """
        Initialize the object.

        Args:
            image_path (str): Mask image file path.
        """

        # Initialize base class.
        #
        super().__init__('Invalid mask image target file path: \'{path}\'.'.format(path=image_path))

        # Store custom data.
        #
        self.image_path = image_path

#----------------------------------------------------------------------------------------------------

class NonLabelImageTypeError(DigitalPathologyImageError):
    """Raise when the format of the image is not suitable for region outlining."""

    def __init__(self, image_dtype, image_coding, image_path):
        """
        Initialize the object.

        Args:
            image_dtype (type): Image data type
            image_coding (str): Image color coding.
            image_path (str): Image file path.
        """

        # Initialize base class.
        #
        super().__init__('The image with {dtype} data type and {coding} color coding is not suitable for region outlining: \'{path}\'.'.format(dtype=image_dtype,
                                                                                                                                               coding=image_coding,
                                                                                                                                               path=image_path))

        # Store custom data.
        #
        self.image_dtype = image_dtype
        self.image_coding = image_coding
        self.image_path = image_path

#----------------------------------------------------------------------------------------------------

class LabelMapConversionOrderMismatchError(DigitalPathologyImageError):
    """Raise when the configured label map does not match conversion order list."""

    def __init__(self, annotation_path, configured_groups, configured_conversion_order):
        """
        Initialize the object.

        Args:
            annotation_path (str): Annotation file path.
            configured_groups (list): Configured annotation groups.
            configured_conversion_order (list): Configured conversion order.
        """

        # Initialize base class.
        #
        message = 'The configured {map} group of annotations to convert does not match the {config} group conversion order in annotation: \'{path}\'.'
        super().__init__(message.format(map=configured_groups, config=configured_conversion_order, path=annotation_path))

        # Store custom data.
        #
        self.annotation_path = annotation_path
        self.configured_groups = configured_groups
        self.configured_conversion_order = configured_conversion_order

#----------------------------------------------------------------------------------------------------

class InvalidFileInvalidAnnotationCoordinateListError(DigitalPathologyImageError):
    """Raise a when an annotation contains an invalid number of points."""

    def __init__(self, coordinate_count, annotation_type, annotation_path):
        """
        Initialize the object.

        Args:
            coordinate_count (int): Number of coordinates.
            annotation_type (str): Annotation type.
            annotation_path (str): Annotation file path.
        """

        # Initialize base class.
        #
        super().__init__('Invalid annotation of type \'{type_id}\' with {count} points, in annotation: \'{path}\'.'.format(type_id=annotation_type, count=coordinate_count, path=annotation_path))

        # Store custom data.
        #
        self.coordinate_count = coordinate_count
        self.annotation_type = annotation_type
        self.annotation_path = annotation_path

#----------------------------------------------------------------------------------------------------

class InvalidCoordinateListError(DigitalPathologyImageError):
    """Raise a when an annotation is added with invalid coordinate list."""

    def __init__(self, coordinate_list):
        """
        Initialize the object.

        Args:
            coordinate_list (np.ndarray, list): Coordinate list.
        """

        # Initialize base class.
        #
        super().__init__('Invalid annotation coordinate list format.')

        # Store custom data.
        #
        self.coordinate_list = coordinate_list

#----------------------------------------------------------------------------------------------------

class InvalidAnnotationCoordinateListError(DigitalPathologyImageError):
    """Raise a when an annotation is added with invalid number of points."""

    def __init__(self, coordinate_count, annotation_type):
        """
        Initialize the object.

        Args:
            coordinate_count (int): Number of coordinates.
            annotation_type (str): Annotation type
        """

        # Initialize base class.
        #
        super().__init__('The {count} available point count is invalid for annotation of type \'{type_id}\'.'.format(count=coordinate_count, type_id=annotation_type))

        # Store custom data.
        #
        self.coordinate_count = coordinate_count
        self.annotation_type = annotation_type

#----------------------------------------------------------------------------------------------------

class UnknownAnnotationTypeError(DigitalPathologyImageError):
    """Raise when the an annotation of unknown type is added."""

    def __init__(self, annotation_type):
        """
        Initialize the object.

        Args:
            annotation_type (str): Annotation type.
        """

        # Initialize base class.
        #
        super().__init__('Unknown annotation type \'{id}\'.'.format(id=annotation_type))

        # Store custom data.
        #
        self.annotation_type = annotation_type

#----------------------------------------------------------------------------------------------------

class InvalidAnnotationIndexError(DigitalPathologyImageError):
    """Raise when an annotation index in an annotation object is invalid."""

    def __init__(self, available_annotations, annotation_index):
        """
        Initialize the object.

        Args:
            available_annotations (int): Number of available annotations.
            annotation_index (int): Annotation index.
        """

        # Initialize base class.
        #
        super().__init__('Invalid annotation index: {index}, with only {available} annotations.'.format(index=annotation_index, available=available_annotations))

        # Store custom data.
        #
        self.available_annotations = available_annotations
        self.annotation_index = annotation_index
#----------------------------------------------------------------------------------------------------

class UnknownAnnotationGroupError(DigitalPathologyImageError):
    """Raise when an annotation group name in an annotation object is invalid."""

    def __init__(self, available_groups, group_name):
        """
        Initialize the object.

        Args:
            available_groups (list): List of available groups.
            group_name (str): Group name.
        """

        # Initialize base class.
        #
        super().__init__('Invalid annotation group: \'{name}\', with available {available} groups.'.format(name=group_name, available=available_groups))

        # Store custom data.
        #
        self.available_groups = available_groups
        self.group_name = group_name

#----------------------------------------------------------------------------------------------------

class InvalidAnnotationColorError(DigitalPathologyImageError):
    """Raise when an annotation color has invalid format."""

    def __init__(self, annotation_color):
        """
        Initialize the object.

        Args:
            annotation_color (tuple): Annotation color.
        """

        # Initialize base class.
        #
        super().__init__('Invalid annotation color format.')

        # Store custom data.
        #
        self.annotation_color = annotation_color

#----------------------------------------------------------------------------------------------------

class MissingPlainImageContentError(DigitalPathologyImageError):
    """Raise when the content of the plain image to write is missing."""

    def __init__(self, image_path):
        """
        Initialize the object.

        Args:
            image_path (str): Path of the image.
        """

        # Initialize base class.
        #
        super().__init__('Missing content for plain image: \'{path}\'.'.format(path=image_path))

        # Store custom data.
        #
        self.image_path = image_path

#----------------------------------------------------------------------------------------------------

class EmptyPlainImageContentError(DigitalPathologyImageError):
    """Raise when the set content for plain image is None."""

    def __init__(self):
        """Initialize the object."""

        # Initialize base class.
        #
        super().__init__('Empty content for plain image.')

#----------------------------------------------------------------------------------------------------

class InvalidContentShapeError(DigitalPathologyImageError):
    """Raise when the set content of the plain image has invalid shape."""

    def __init__(self, content_shape):
        """
        Initialize the object.

        Args:
            content_shape (tuple): Content shape.
        """

        # Initialize base class.
        #
        super().__init__('The configured content has invalid {shape} shape.'.format(shape=content_shape))

        # Store custom data.
        #
        self.content_shape = content_shape
