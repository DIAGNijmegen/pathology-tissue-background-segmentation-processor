"""
Image processing related errors.
"""

from . import errorbase as dpterrorbase

#----------------------------------------------------------------------------------------------------

class DigitalPathologyProcessingError(dpterrorbase.DigitalPathologyError):
    """Error base class for all image processing errors."""

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

class EmptyAnnotationGroupsError(DigitalPathologyProcessingError):
    """Raise when all the selected annotation group are empty and empty groups are not allowed."""

    def __init__(self, group_names):
        """
        Initialize the object.

        Args:
            group_names (list): Empty annotation group names.
        """

        # Initialize base class.
        #
        super().__init__('No annotation in the selected groups: {groups}.'.format(groups=group_names))

        # Store custom data.
        #
        self.group_names = group_names

#----------------------------------------------------------------------------------------------------

class UnknownAnnotationGroupsError(DigitalPathologyProcessingError):
    """Raise when there are unknown annotation groups present and strict mode is enabled."""

    def __init__(self, group_names):
        """
        Initialize the object.

        Args:
            group_names (set): Unknown annotation group names.
        """

        # Initialize base class.
        #
        super().__init__('Unknown annotation groups: {groups}.'.format(groups=group_names))

        # Store custom data.
        #
        self.group_names = group_names

#----------------------------------------------------------------------------------------------------

class NoMatchingLevelInTileImageError(DigitalPathologyProcessingError):
    """Raise when no matching level found in the tile template image for the mask for normalization."""

    def __init__(self, tile_path, mask_path):
        """
        Initialize the object.

        Args:
            tile_path (str): Tile image path.
            mask_path (str): Mask image path.
        """

        # Initialize base class.
        #
        super().__init__('No matching level found in the template image for the mask.')

        # Store custom data.
        #
        self.tile_path = tile_path
        self.mask_path = mask_path

#----------------------------------------------------------------------------------------------------

class UnknownImageArithmeticOperandError(DigitalPathologyProcessingError):
    """Raise when the image arithmetic operand is unknown."""

    def __init__(self, operand):
        """
        Initialize the object.

        Args:
            operand (str): Image arithmetic operand.
        """

        # Initialize base class.
        #
        super().__init__('Unknown image arithmetic operand: \'{operand}\'.'.format(operand=operand))

        # Store custom data.
        #
        self.operand = operand

#----------------------------------------------------------------------------------------------------

class ImageShapeMismatchError(DigitalPathologyProcessingError):
    """Raise when the image shapes do not match."""

    def __init__(self, left_path, left_shape, right_path, right_shape):
        """
        Initialize the object.

        Args:
            left_path (str): Left image path.
            left_shape (tuple): Left image shape.
            right_path (str): Right image path.
            right_shape (tuple): Right image shape.
        """

        # Initialize base class.
        #
        super().__init__('Image shape mismatch: {left} - {right}.'.format(left=left_shape, right=right_shape))

        # Store custom data.
        #
        self.left_path = left_path
        self.left_shape = left_shape
        self.right_path = right_path
        self.right_shape = right_shape

#----------------------------------------------------------------------------------------------------

class ArrayShapeMismatch(DigitalPathologyProcessingError):
    """Raise when the array shapes do not match."""

    def __init__(self, left_shape, right_shape):
        """
        Initialize the object.

        Args:
            left_shape (tuple): Left array shape.
            right_shape (tuple): Right array shape.
        """

        # Initialize base class.
        #
        super().__init__('Array shape mismatch: {left} - {right}.'.format(left=left_shape, right=right_shape))

        # Store custom data.
        #
        self.left_shape = left_shape
        self.right_shape = right_shape

#----------------------------------------------------------------------------------------------------

class InvalidPixelSpacingValueError(DigitalPathologyProcessingError):
    """Raise when both image level and pixel spacing are None."""

    def __init__(self):
        """Initialize the object."""

        # Initialize base class.
        #
        super().__init__('Both image level and pixel spacing are None.')

#----------------------------------------------------------------------------------------------------

class ReferenceTemplateImageShapeMismatchError(DigitalPathologyProcessingError):
    """Raise when the compared reference and template image shapes do not match."""

    def __init__(self, reference_path, reference_shape, template_path, template_shape):
        """
        Initialize the object.

        Args:
            reference_path (str): Reference image path.
            reference_shape (tuple): Reference shape.
            template_path (str): Template image path.
            template_shape (tuple): Template shape.
        """

        # Initialize base class.
        #
        super().__init__('The {template} template image does not match the {reference} reference image shape.'.format(template=template_shape, reference=reference_shape))

        # Store custom data.
        #
        self.reference_path = reference_path
        self.reference_shape = reference_shape
        self.template_path = template_path
        self.template_shape = template_shape

#----------------------------------------------------------------------------------------------------

class MissingModelFileError(DigitalPathologyProcessingError):
    """Raise when the network model file to load is non-existent."""

    def __init__(self, model_path):
        """
        Initialize the object.

        Args:
             model_path (str): Model file path.
        """

        # Initialize base class.
        #
        super().__init__('Missing model file: {path}.'.format(path=model_path))

        # Store custom data.
        #
        self.model_path = model_path

#----------------------------------------------------------------------------------------------------

class UnknownImageNormalizationModeError(DigitalPathologyProcessingError):
    """Raise when the image normalization mode is unknown."""

    def __init__(self, comparison_mode):
        """
        Initialize the object.

        Args:
             comparison_mode (str): Image comparison mode.
        """

        # Initialize base class.
        #
        super().__init__('Unknown image comparison mode: \'{mode}\'.'.format(mode=comparison_mode))

        # Store custom data.
        #
        self.comparison_mode = comparison_mode

#----------------------------------------------------------------------------------------------------

class InvalidConfigurationError(DigitalPathologyProcessingError):
    """Raise when the steps enabled with binary input in network inference, but the network output is non-binary.."""

    def __init__(self, steps):
        """
        Initialize the object.

        Args:
            steps (list): List of steps enabled.
        """

        # Initialize base class.
        #
        super().__init__('Non-binary network output with {count} processing steps that requires binary input.'.format(count=sum(steps)))

        # Store custom data.
        #
        self.steps = steps

#----------------------------------------------------------------------------------------------------

class InvalidColorPaletteError(DigitalPathologyProcessingError):
    """Raise when the palette for mask image coloring is invalid."""

    def __init__(self, palette):
        """
        Initialize the object.

        Args:
             palette (str): Image comparison mode.
        """

        # Initialize base class.
        #
        super().__init__('Invalid palette: {palette}.'.format(palette=palette))

        # Store custom data.
        #
        self.palette = palette

#----------------------------------------------------------------------------------------------------

class NoMatchingLevelInMaskImageError(DigitalPathologyProcessingError):
    """Raise when no matching level found in the mask image for the RGB image at the given level for preview."""

    def __init__(self, mask_path, image_path, image_level):
        """
        Initialize the object.

        Args:
            mask_path (str): Mask image path.
            image_path (str): Image path.
            image_level (int): Image level.
        """

        # Initialize base class.
        #
        super().__init__('No matching level found in the mask image for the image at level {level}.'.format(level=image_level))

        # Store custom data.
        #
        self.mask_path = mask_path
        self.image_path = image_path
        self.image_level = image_level

#----------------------------------------------------------------------------------------------------

class ThresholdDimensionImageChannelCountMismatchError(DigitalPathologyProcessingError):
    """Raise when the dimensions of the threshold does not match the image channels."""

    def __init__(self, threshold_dimensions, image_channels):
        """
        Initialize the object.

        Args:
            threshold_dimensions (int): Threshold dimension count.
            image_channels (int): Image Channel count.
        """

        # Initialize base class.
        #
        super().__init__('Threshold length ({thresholds}) should be larger or equal to number of patch channels ({channels}).'.format(thresholds=threshold_dimensions, channels=image_channels))

        # Store custom data.
        #
        self.threshold_dimensions = threshold_dimensions
        self.image_channels = image_channels

#----------------------------------------------------------------------------------------------------

class InvalidDilationDistanceError(DigitalPathologyProcessingError):
    """Raise when the dilation distance is invalid."""

    def __init__(self, dilation_distance):
        """
        Initialize the object.

        Args:
            dilation_distance (float): Dilation distance.
        """

        # Initialize base class.
        #
        super().__init__('Invalid dilation distance: {distance} um.'.format(distance=dilation_distance))

        # Store custom data.
        #
        self.dilation_distance = dilation_distance

#----------------------------------------------------------------------------------------------------

class NonMonochromeInputImageError(DigitalPathologyProcessingError):
    """Raise when the input image is not monochrome as expected."""

    def __init__(self, image_path, image_coding):
        """
        Initialize the object.

        Args:
            image_path (str): Image file path.
            image_coding (str): Image color coding.
        """

        # Initialize base class.
        #
        super().__init__('The image is not monochrome: {path}.'.format(path=image_path))

        # Store custom data.
        #
        self.image_path = image_path
        self.image_coding = image_coding

#----------------------------------------------------------------------------------------------------

class NonIntegralImageDataTypeError(DigitalPathologyProcessingError):
    """Raise when the input does not have an integral data type as expected."""

    def __init__(self, image_path, image_dtype):
        """
        Initialize the object.

        Args:
            image_path (str): Image file path.
            image_dtype (type): Image data type.
        """

        # Initialize base class.
        #
        super().__init__('The image data type is not integral: {path}.'.format(path=image_path))

        # Store custom data.
        #
        self.image_path = image_path
        self.image_dtype = image_dtype

#----------------------------------------------------------------------------------------------------

class InvalidInterpolationOrderError(DigitalPathologyProcessingError):
    """Raise when the interpolation order is out of [0, 5] bounds."""

    def __init__(self, interpolation_order):
        """
        Initialize the object.

        Args:
            interpolation_order (int): Interpolation order.
        """

        # Initialize base class.
        #
        super().__init__('Interpolation order {order} is out of [0, 5] bounds.'.format(order=interpolation_order))

        # Store custom data.
        #
        self.interpolation_order = interpolation_order

#----------------------------------------------------------------------------------------------------

class AsymmetricZoomFactorError(DigitalPathologyProcessingError):
    """Raise when the height and width zoom factors are different."""

    def __init__(self, height_factor, width_factor):
        """
        Initialize the object.

        Args:
            height_factor (float): Height zoom factor.
            width_factor (float): Width zoom factor.
        """

        # Initialize base class.
        #
        super().__init__('Asymmetric zoom factors: {height} - {width}.'.format(height=height_factor, width=width_factor))

        # Store custom data.
        #
        self.height_factor = height_factor
        self.width_factor = width_factor

#----------------------------------------------------------------------------------------------------

class InvalidZoomFactorError(DigitalPathologyProcessingError):
    """Raise when the zoom factor is invalid."""

    def __init__(self, zoom_factor):
        """
        Initialize the object.

        Args:
            zoom_factor (float): Zoom factor.
        """

        # Initialize base class.
        #
        super().__init__('Invalid zoom factor: {factor}.'.format(factor=zoom_factor))

        # Store custom data.
        #
        self.zoom_factor = zoom_factor

#----------------------------------------------------------------------------------------------------

class InvalidZoomTargetShapeError(DigitalPathologyProcessingError):
    """Raise when the target shape to zoom to is invalid."""

    def __init__(self, target_shape):
        """
        Initialize the object.

        Args:
            target_shape (tuple): Target shape for zooming
        """

        # Initialize base class.
        #
        super().__init__('Invalid target shape for zooming: {shape}.'.format(shape=target_shape))

        # Store custom data.
        #
        self.target_shape = target_shape
