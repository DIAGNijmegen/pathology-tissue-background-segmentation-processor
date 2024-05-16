"""
Augmentation related errors.
"""

from . import errorbase as dpterrorbase

#----------------------------------------------------------------------------------------------------

class DigitalPathologyAugmentationError(dpterrorbase.DigitalPathologyError):
    """Error base class for all augmentation errors."""

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

class InvalidBlurSigmaRangeError(DigitalPathologyAugmentationError):
    """Raise when the range of sigma for Gaussian blurring is invalid."""

    def __init__(self, sigma_range):
        """
        Initialize the object.

        Args:
            sigma_range (list, tuple): Sigma range.
        """

        # Initialize base class.
        #
        super().__init__('Invalid Gaussian blur sigma range: {interval}.'.format(interval=sigma_range))

        # Store custom data.
        #
        self.sigma_range = sigma_range

#----------------------------------------------------------------------------------------------------

class InvalidAdditiveGaussianNoiseSigmaRangeError(DigitalPathologyAugmentationError):
    """Raise when the range of sigma for Gaussian noise is invalid."""

    def __init__(self, sigma_range):
        """
        Initialize the object.

        Args:
            sigma_range (list, tuple): Sigma range.
        """

        # Initialize base class.
        #
        super().__init__('Invalid additive Gaussian noise sigma range: {interval}.'.format(interval=sigma_range))

        # Store custom data.
        #
        self.sigma_range = sigma_range

#----------------------------------------------------------------------------------------------------

class InvalidScalingRangeError(DigitalPathologyAugmentationError):
    """Raise when the range of sigma for scaling is invalid."""

    def __init__(self, scaling_range):
        """
        Initialize the object.

        Args:
            scaling_range (list, tuple): Scaling range.
        """

        # Initialize base class.
        #
        super().__init__('Invalid scaling sigma range: {interval}.'.format(interval=scaling_range))

        # Store custom data.
        #
        self.scaling_range = scaling_range

#----------------------------------------------------------------------------------------------------

class InvalidScalingInterpolationOrderError(DigitalPathologyAugmentationError):
    """Raise when the interpolation order for scaling is invalid."""

    def __init__(self, interpolation_order):
        """
        Initialize the object.

        Args:
            interpolation_order (int): Interpolation order.
        """

        # Initialize base class.
        #
        super().__init__('Invalid scaling interpolation order: {order}.'.format(order=interpolation_order))

        # Store custom data.
        #
        self.interpolation_order = interpolation_order

#----------------------------------------------------------------------------------------------------

class InvalidElasticInterpolationOrderError(DigitalPathologyAugmentationError):
    """Raise when the interpolation order for elastic transformation is invalid."""

    def __init__(self, interpolation_order):
        """
        Initialize the object.

        Args:
            interpolation_order (int): Interpolation order.
        """

        # Initialize base class.
        #
        super().__init__('Invalid elastic transformation interpolation order: {order}.'.format(order=interpolation_order))

        # Store custom data.
        #
        self.interpolation_order = interpolation_order

#----------------------------------------------------------------------------------------------------

class InvalidContrastSigmaRangeError(DigitalPathologyAugmentationError):
    """Raise when the contrast adjustment sigma range is not valid."""

    def __init__(self, sigma_range):
        """
        Initialize the object.

        Args:
            sigma_range (list, tuple): Range for contrast adjustment.
        """

        # Initialize base class.
        #
        super().__init__('Invalid contrast adjustment sigma range: {interval}.'.format(interval=sigma_range))

        # Store custom data.
        #
        self.sigma_range = sigma_range

#----------------------------------------------------------------------------------------------------

class InvalidHaematoxylinSigmaRangeError(DigitalPathologyAugmentationError):
    """Raise when the sigma range for Haematoxylin channel adjustment is not valid."""

    def __init__(self, sigma_range):
        """
        Initialize the object.

        Args:
            sigma_range (list, tuple): Sigma range.
        """

        # Initialize base class.
        #
        super().__init__('Invalid haematoxylin channel adjustment sigma range: {interval}.'.format(interval=sigma_range))

        # Store custom data.
        #
        self.sigma_range = sigma_range

#----------------------------------------------------------------------------------------------------

class InvalidHaematoxylinBiasRangeError(DigitalPathologyAugmentationError):
    """Raise when the bias range for Haematoxylin channel adjustment is not valid."""

    def __init__(self, bias_range):
        """
        Initialize the object.

        Args:
            bias_range (list, tuple): Bias range.
        """

        # Initialize base class.
        #
        super().__init__('Invalid haematoxylin channel adjustment bias range: {interval}.'.format(interval=bias_range))

        # Store custom data.
        #
        self.bias_range = bias_range

#----------------------------------------------------------------------------------------------------

class InvalidEosinSigmaRangeError(DigitalPathologyAugmentationError):
    """Raise when the sigma range for Eosin channel adjustment is not valid."""

    def __init__(self, sigma_range):
        """
        Initialize the object.

        Args:
            sigma_range (list, tuple): Sigma range.
        """

        # Initialize base class.
        #
        super().__init__('Invalid eosin channel adjustment sigma range: {interval}.'.format(interval=sigma_range))

        # Store custom data.
        #
        self.sigma_range = sigma_range

#----------------------------------------------------------------------------------------------------

class InvalidEosinBiasRangeError(DigitalPathologyAugmentationError):
    """Raise when the bias range for Eosin channel adjustment is not valid."""

    def __init__(self, bias_range):
        """
        Initialize the object.

        Args:
            bias_range (list, tuple): Bias range.
        """

        # Initialize base class.
        #
        super().__init__('Invalid eosin channel adjustment bias range: {interval}.'.format(interval=bias_range))

        # Store custom data.
        #
        self.bias_range = bias_range

#----------------------------------------------------------------------------------------------------

class InvalidDabSigmaRangeError(DigitalPathologyAugmentationError):
    """Raise when the sigma range for DAB channel adjustment is not valid."""

    def __init__(self, sigma_range):
        """
        Initialize the object.

        Args:
            sigma_range (list, tuple): Sigma range.
        """

        # Initialize base class.
        #
        super().__init__('Invalid DAB channel adjustment sigma range: {interval}.'.format(interval=sigma_range))

        # Store custom data.
        #
        self.sigma_range = sigma_range

#----------------------------------------------------------------------------------------------------

class InvalidDabBiasRangeError(DigitalPathologyAugmentationError):
    """Raise when the bias range for DAB channel adjustment is not valid."""

    def __init__(self, bias_range):
        """
        Initialize the object.

        Args:
            bias_range (list, tuple): Bias range.
        """

        # Initialize base class.
        #
        super().__init__('Invalid DAB channel adjustment bias range: {interval}.'.format(interval=bias_range))

        # Store custom data.
        #
        self.bias_range = bias_range

#----------------------------------------------------------------------------------------------------

class InvalidCutoffRangeError(DigitalPathologyAugmentationError):
    """Raise when the cutoff range for HED color augmentation is not valid."""

    def __init__(self, cutoff_range):
        """
        Initialize the object.

        Args:
            cutoff_range (list, tuple): Cutoff range.
        """

        # Initialize base class.
        #
        super().__init__('Invalid HED color augmentation cutoff range: {interval}.'.format(interval=cutoff_range))

        # Store custom data.
        #
        self.cutoff_range = cutoff_range

#----------------------------------------------------------------------------------------------------

class InvalidHueSigmaRangeError(DigitalPathologyAugmentationError):
    """Raise when the sigma range for Hue channel adjustment is not valid."""

    def __init__(self, sigma_range):
        """
        Initialize the object.

        Args:
            sigma_range (list, tuple): Sigma range.
        """

        # Initialize base class.
        #
        super().__init__('Invalid hue channel adjustment sigma range: {interval}.'.format(interval=sigma_range))

        # Store custom data.
        #
        self.sigma_range = sigma_range

#----------------------------------------------------------------------------------------------------

class InvalidSaturationSigmaRangeError(DigitalPathologyAugmentationError):
    """Raise when the sigma range for Saturation channel adjustment is not valid."""

    def __init__(self, sigma_range):
        """
        Initialize the object.

        Args:
            sigma_range (list, tuple): Sigma range.
        """

        # Initialize base class.
        #
        super().__init__('Invalid saturation channel adjustment sigma range: {interval}.'.format(interval=sigma_range))

        # Store custom data.
        #
        self.sigma_range = sigma_range

#----------------------------------------------------------------------------------------------------

class InvalidBrightnessSigmaRangeError(DigitalPathologyAugmentationError):
    """Raise when the sigma range for Brightness channel adjustment is not valid."""

    def __init__(self, sigma_range):
        """
        Initialize the object.

        Args:
            sigma_range (list, tuple): Sigma range.
        """

        # Initialize base class.
        #
        super().__init__('Invalid brightness channel adjustment sigma range: {interval}.'.format(interval=sigma_range))

        # Store custom data.
        #
        self.sigma_range = sigma_range

#----------------------------------------------------------------------------------------------------

class InvalidRotationRepetitionListError(DigitalPathologyAugmentationError):
    """Raise when the list for 90 degree rotation repetition is invalid."""

    def __init__(self, k_list):
        """
        Initialize the object.

        Args:
            k_list (list): List of 90 degree rotation repetition times.
        """

        # Initialize base class.
        #
        super().__init__('Invalid 90 degree rotation repetition list: {repetitions}.'.format(repetitions=k_list))

        # Store custom data.
        #
        self.k_list = k_list

#----------------------------------------------------------------------------------------------------

class InvalidFlipListError(DigitalPathologyAugmentationError):
    """Raise when the list for flip augmentation is invalid."""

    def __init__(self, f_list):
        """
        Initialize the object.

        Args:
            f_list (list): List of 90 degree rotation repetition times.
        """

        # Initialize base class.
        #
        super().__init__('Invalid flip augmentation list: {flips}.'.format(flips=f_list))

        # Store custom data.
        #
        self.f_list = f_list

#----------------------------------------------------------------------------------------------------

class InvalidFlipMode(DigitalPathologyAugmentationError):
    """Raise when the flip mode is not supported."""

    def __init__(self, f_mode):
        """
        Initialize the object.

        Args:
            f_mode (str): Flip mode.
        """

        # Initialize base class.
        #
        super().__init__('Invalid flip mode selected: {flip_mode}.'.format(flip_mode=f_mode))

        # Store custom data.
        #
        self.f_mode = f_mode

#----------------------------------------------------------------------------------------------------

class InvalidElasticSigmaIntervalError(DigitalPathologyAugmentationError):
    """Raise when the interval of sigma for elastic deformation is invalid."""

    def __init__(self, sigma_interval):
        """
        Initialize the object.

        Args:
            sigma_interval (list, tuple): Sigma interval.
        """

        # Initialize base class.
        #
        super().__init__('Invalid elastic deformation sigma interval: {interval}.'.format(interval=sigma_interval))

        # Store custom data.
        #
        self.sigma_interval = sigma_interval

#----------------------------------------------------------------------------------------------------

class InvalidElasticAlphaIntervalError(DigitalPathologyAugmentationError):
    """Raise when the interval of alpha for elastic deformation is invalid."""

    def __init__(self, alpha_interval):
        """
        Initialize the object.

        Args:
            alpha_interval (list, tuple): Alpha interval.
        """

        # Initialize base class.
        #
        super().__init__('Invalid elastic deformation alpha interval: {interval}.'.format(interval=alpha_interval))

        # Store custom data.
        #
        self.alpha_interval = alpha_interval

#----------------------------------------------------------------------------------------------------

class InvalidElasticMapCountError(DigitalPathologyAugmentationError):
    """Raise when the number of elastic deformation maps to precalculate is invalid."""

    def __init__(self, map_count):
        """
        Initialize the object.

        Args:
            map_count (int): Amount of deformation maps to precalculate.
        """

        # Initialize base class.
        #
        super().__init__('Invalid count of elastic deformation maps to precalculate: {count}.'.format(count=map_count))

        # Store custom data.
        #
        self.map_count = map_count

#----------------------------------------------------------------------------------------------------

class EmptyAugmentationGroupListError(DigitalPathologyAugmentationError):
    """Raise when the list of augmentation groups is empty."""

    def __init__(self):
        """Initialize the object."""

        # Initialize base class.
        #
        super().__init__('The list of augmentation groups is empty.')

#----------------------------------------------------------------------------------------------------

class AugmentationGroupAlreadyExistsError(DigitalPathologyAugmentationError):
    """Raise when the augmentation group already exists when added to the pool."""

    def __init__(self, group_id):
        """
        Initialize the object.

        Args:
            group_id (str): Group identifier.
        """

        # Initialize base class.
        #
        super().__init__('Augmentation group already exists: {group}.'.format(group=group_id))

        # Store custom data.
        #
        self.group_id = group_id

#----------------------------------------------------------------------------------------------------

class UnknownAugmentationGroupError(DigitalPathologyAugmentationError):
    """Raise when the augmentation group is unknown."""

    def __init__(self, group_id):
        """
        Initialize the object.

        Args:
            group_id (str): Group identifier.
        """

        # Initialize base class.
        #
        super().__init__('Unknown augmentation group: {group}.'.format(group=group_id))

        # Store custom data.
        #
        self.group_id = group_id

#----------------------------------------------------------------------------------------------------

class InvalidAugmentationRatioError(DigitalPathologyAugmentationError):
    """Raise when the ratio of the augmentation selection in its group is invalid."""

    def __init__(self, ratio):
        """
        Initialize the object.

        Args:
            ratio (float): Ratio for augmentation selection.
        """

        # Initialize base class.
        #
        super().__init__('Invalid augmentation selection ratio: {ratio}.'.format(ratio=ratio))

        # Store custom data.
        #
        self.ratio = ratio

#----------------------------------------------------------------------------------------------------

class MissingAugmentationRandomizationError(DigitalPathologyAugmentationError):
    """Raise when the augmentation selection in pool is nor randomized."""

    def __init__(self):
        """Initialize the object."""

        # Initialize base class.
        #
        super().__init__('The augmentation selection in pool is not randomized.')

#----------------------------------------------------------------------------------------------------

class AugmentationPoolRandomizationBeforeDistribution(DigitalPathologyAugmentationError):
    """Raise when the augmentation selection is randomized without distribution."""

    def __init__(self):
        """Initialize the object."""

        # Initialize base class.
        #
        super().__init__('The augmentation pool is randomized without distribution.')

#----------------------------------------------------------------------------------------------------

class EmptyAugmentationGroupError(DigitalPathologyAugmentationError):
    """Raise when the augmentation group is empty."""

    def __init__(self, augmentation_groups):
        """
        Initialize the object.

        Args:
            augmentation_groups (list): Empty augmentation groups.
        """

        # Initialize base class.
        #
        super().__init__('Empty augmentation group: {groups}.'.format(groups=augmentation_groups))

        # Store custom data.
        #
        self.augmentation_groups = augmentation_groups

#----------------------------------------------------------------------------------------------------

class PatchCroppingError(DigitalPathologyAugmentationError):
    """Raise when the target shape for cropping is smaller than the patch to crop itself."""

    def __init__(self, patch_shape, target_shape):
        """
        Initialize the object.

        Args:
            patch_shape (tuple): Shape of the patch to crop.
            target_shape (tuple): Target shape for cropping.
        """

        # Initialize base class.
        #
        super().__init__('Target {target} shape for cropping is smaller than the {patch} patch shape to crop.'.format(target=target_shape, patch=patch_shape))

        # Store custom data.
        #
        self.patch_shape = patch_shape
        self.target_shape = target_shape

#----------------------------------------------------------------------------------------------------

class BatchCroppingError(DigitalPathologyAugmentationError):
    """Raise when the target shape for cropping is smaller than the patch to crop itself."""

    def __init__(self, batch_shape, target_shape):
        """
        Initialize the object.

        Args:
            batch_shape (tuple): Shape of the batch to crop.
            target_shape (tuple): Target shape for cropping.
        """

        # Initialize base class.
        #
        super().__init__('Target {target} shape for cropping is smaller than the {patch} batch shape to crop.'.format(target=target_shape, patch=batch_shape))

        # Store custom data.
        #
        self.batch_shape = batch_shape
        self.target_shape = target_shape

#----------------------------------------------------------------------------------------------------

class MissingTargetShapeForLevelError(DigitalPathologyAugmentationError):
    """Raise when a target shape for cropping is missing for a level."""

    def __init__(self, batch_levels, crop_levels):
        """
        Initialize the object.

        Args:
            batch_levels (list): List of levels in a batch.
            crop_levels (list): List of levels in the target shapes.
        """

        # Initialize base class.
        #
        super().__init__('Batch {batch} and cropping target shape {crop} levels mismatch.'.format(batch=batch_levels, crop=crop_levels))

        # Store custom data.
        #
        self.batch_levels = batch_levels
        self.crop_levels = crop_levels
