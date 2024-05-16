"""
Weight map calculation errors.
"""

from . import errorbase as dpterrorbase

#----------------------------------------------------------------------------------------------------

class DigitalPathologyWeightMappingError(dpterrorbase.DigitalPathologyError):
    """Error base class for all weight mapping errors."""

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

class InvalidClippingRangeError(DigitalPathologyWeightMappingError):
    """Raise when the clipping is enabled and the range is invalid."""

    def __init__(self, clip_min, clip_max):
        """
        Initialize the object.

        Args:
            clip_min (float): Minimum value for clipping.
            clip_max (float): Maximum value for clipping.
        """

        # Initialize base class.
        #
        super().__init__('Invalid clipping range: [{clip_min}, {clip_max}].'.format(clip_min=clip_min, clip_max=clip_max))

        # Store custom data.
        #
        self.clip_min = clip_min
        self.clip_max = clip_max

#----------------------------------------------------------------------------------------------------

class WeightMappingConfigurationError(DigitalPathologyWeightMappingError):
    """Raise when weight mapping is configured without label patch extraction."""

    def __init__(self, label_mode):
        """
        Initialize the object.

        Args:
            label_mode (str): Label mode.
        """

        # Initialize base class.
        #
        super().__init__('Weight configured with incompatible label mode: {mode}.'.format(mode=label_mode))

        # Store custom data.
        #
        self.label_mode = label_mode

#----------------------------------------------------------------------------------------------------

class MissingLabelMapperForWeightMapperError(DigitalPathologyWeightMappingError):
    """Raise when weight mapping is configured without label mapping."""

    def __init__(self):
        """Initialize the object."""

        # Initialize base class.
        #
        super().__init__('Label mapper is not configured for the weight mapper.')

#----------------------------------------------------------------------------------------------------

class WeightMapperLabelMapperClassesMismatchError(DigitalPathologyWeightMappingError):
    """Raise when network labels known by the label mapper does not match the network labels known by the weight mapper."""

    def __init__(self, weight_mapper_classes, label_mapper_classes):
        """
        Initialize the object.

        Args:
            weight_mapper_classes (int): Classes of the weight mapper.
            label_mapper_classes (int): Classes of the label mapper.
        """

        # Initialize base class.
        #
        super().__init__('The {weight} classes of the weight mapper does not match the {label} classes of the label mapper.'.format(weight=weight_mapper_classes, label=label_mapper_classes))

        # Store custom data.
        #
        self.weight_mapper_classes = weight_mapper_classes
        self.label_mapper_classes = label_mapper_classes

#----------------------------------------------------------------------------------------------------

class MissingPatchMapperWithBatchMapper(DigitalPathologyWeightMappingError):
    """"Raise when there is a batch weight mapper without a patch weight mapper."""

    def __init__(self):
        """
        Initialize the object.
        """

        # Initialize base class.
        #
        super().__init__('A batch weight mapper can not be used without a patch weight mapper.')
