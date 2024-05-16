"""
Configuration related errors.
"""

from . import errorbase as dpterrorbase

#----------------------------------------------------------------------------------------------------

class DigitalPathologyConfigError(dpterrorbase.DigitalPathologyError):
    """Error base class for all configuration errors."""

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

class EmptyPixelSpacingListError(DigitalPathologyConfigError):
    """Raise when the list of used pixel spacings is empty."""

    def __init__(self):
        """Initialize the object."""

        # Initialize base class.
        #
        super().__init__('Empty pixel spacing list.')

#----------------------------------------------------------------------------------------------------

class InvalidPixelSpacingInPatchShapesError(DigitalPathologyConfigError):
    """Raise when the pixel spacing is invalid."""

    def __init__(self, patch_shapes):
        """
        Initialize the object.

        Args:
            patch_shapes (dict): Pixel spacing to patch shape dictionary.
        """

        # Initialize base class.
        #
        super().__init__('Invalid pixel spacing {{spacing: (rows, cols)}}: {spacings}.'.format(spacings=patch_shapes))

        # Store custom data.
        #
        self.patch_shapes = patch_shapes

#----------------------------------------------------------------------------------------------------

class EmptyChannelListError(DigitalPathologyConfigError):
    """Raise when the list of used channels is empty."""

    def __init__(self):
        """Initialize the object."""

        # Initialize base class.
        #
        super().__init__('Empty input channel list.')

#----------------------------------------------------------------------------------------------------

class InvalidDimensionOrderError(DigitalPathologyConfigError):
    """Raise when the dimension order definition is invalid."""

    def __init__(self, dimension_order):
        """
        Initialize the object.

        Args:
            dimension_order (str): Dimension order definition.
        """

        # Initialize base class.
        #
        super().__init__('Invalid dimension order: \'{order}\'.'.format(order=dimension_order))

        # Store custom data.
        #
        self.dimension_order = dimension_order

#----------------------------------------------------------------------------------------------------

class InvalidPatchShapeError(DigitalPathologyConfigError):
    """Raise when the patch shape is invalid."""

    def __init__(self, patch_shapes):
        """
        Initialize the object.

        Args:
            patch_shapes (dict): Pixel spacing to patch shape dictionary.
        """

        # Initialize base class.
        #
        super().__init__('Invalid patch shape {{spacing: (rows, cols)}}: {shapes}.'.format(shapes=patch_shapes))

        # Store custom data.
        #
        self.patch_shapes = patch_shapes

#----------------------------------------------------------------------------------------------------

class InvalidMaskPixelSpacingError(DigitalPathologyConfigError):
    """Raise when the mask pixel spacing is invalid."""

    def __init__(self, mask_spacing):
        """
        Initialize the object.

        Args:
            mask_spacing (float): Mask pixel spacing.
        """

        # Initialize base class.
        #
        super().__init__('Invalid mask pixel spacing: {spacing}.'.format(spacing=mask_spacing))

        # Store custom data.
        #
        self.mask_spacing = mask_spacing

#----------------------------------------------------------------------------------------------------

class InvalidPixelSpacingToleranceError(DigitalPathologyConfigError):
    """Raise when the pixel spacing tolerance is invalid."""

    def __init__(self, tolerance):
        """
        Initialize the object.

        Args:
            tolerance (float, None): Pixel spacing tolerance (percentage).
        """

        # Initialize base class.
        #
        super().__init__('Invalid pixel spacing tolerance: {tolerance}.'.format(tolerance=tolerance))

        # Store custom data.
        #
        self.tolerance = tolerance

#----------------------------------------------------------------------------------------------------

class DuplicateChannelError(DigitalPathologyConfigError):
    """Raise when the list of channels contains duplicates."""

    def __init__(self, input_channels):
        """
        Initialize the object.

        Args:
            input_channels (list): Channel list.
        """

        # Initialize base class.
        #
        super().__init__('Duplicate input channel values: {channels}.'.format(channels=input_channels))

        # Store custom data.
        #
        self.input_channels = input_channels

#----------------------------------------------------------------------------------------------------

class PurposeListAndRatioMismatchError(DigitalPathologyConfigError):
    """Raise when the purpose ratios does not match the list of available purposes."""

    def __init__(self, purpose_distribution, available_purposes):
        """
        Initialize the object.

        Args:
            purpose_distribution (dict): Configured purpose distribution.
            available_purposes (list): List of available purposes.
        """

        # Initialize base class.
        #
        super().__init__('The configured {config} distribution does not match the {available} list of available purposes.'.format(config=purpose_distribution, available=available_purposes))

        # Store custom data.
        #
        self.purpose_distribution = purpose_distribution
        self.available_purposes = available_purposes

#----------------------------------------------------------------------------------------------------

class InvalidBatchSizeError(DigitalPathologyConfigError):
    """Raise when the requested batch size is less than 1."""

    def __init__(self, batch_size):
        """
        Initialize the object.

        Args:
            batch_size (int): Batch size.
        """

        # Initialize base class.
        #
        super().__init__('Invalid batch size: {size}.'.format(size=batch_size))

        # Store custom data.
        #
        self.batch_size = batch_size

#----------------------------------------------------------------------------------------------------

class BatchSizeOutOfBoundsError(DigitalPathologyConfigError):
    """Raise when the requested batch size is invalid: either too large or negative."""

    def __init__(self, batch_size, batch_size_bounds):
        """
        Initialize the object.

        Args:
             batch_size (int): Number of patches to load.
             batch_size_bounds (tuple): Limits of the batch size: (min, max)
        """

        # Initialize base class.
        #
        super().__init__('Batch size out of bounds: {size}; [{left}, {right}].'.format(size=batch_size, left=batch_size_bounds[0], right=batch_size_bounds[1]))

        # Store custom data.
        #
        self.batch_size = batch_size
        self.batch_size_bounds = batch_size_bounds

#----------------------------------------------------------------------------------------------------

class UnknownLabelModeError(DigitalPathologyConfigError):
    """Raise when the label generation mode is unknown."""

    def __init__(self, mode):
        """
        Initialize the object.

        Args:
            mode (str): Label generation mode.
        """

        # Initialize base class.
        #
        super().__init__('Unknown label generation mode: \'{mode}\'.'.format(mode=mode))

        # Store custom data.
        #
        self.mode = mode

#----------------------------------------------------------------------------------------------------

class MissingMaskImageError(DigitalPathologyConfigError):
    """Raise when the label generation mode is 'load' but there is not mask image loaded."""

    def __init__(self):
        """Initialize the object."""

        # Initialize base class.
        #
        super().__init__('Label generation mode is set to \'load\' but no mask image is loaded.')

#----------------------------------------------------------------------------------------------------

class InvalidDataFileExtensionError(DigitalPathologyConfigError):
    """Raise when the data file format cannot be derived from the target file extension."""

    def __init__(self, file_path):
        """
        Initialize the object.

        Args:
             file_path (str): Target path.
        """

        # Initialize base class.
        #
        super().__init__('Cannot derive data file format from extension: \'{path}\'.'.format(path=file_path))

        # Store custom data.
        #
        self.file_path = file_path
