"""
This file contains the base class for calculating weight maps for U-Net like networks.
"""

from ..errors import weighterrors as dptweighterrors

#----------------------------------------------------------------------------------------------------

class WeightMapperBase(object):
    """Base class for weight map calculation."""

    def __init__(self, classes, clip_min, clip_max):
        """
        Initialize the object.

        Args:
            classes (int): Number of classes. The input labels are expected in the [0, classes-1] range.
            clip_min (float): Minimum value for clipping.
            clip_max (float): Maximum value for clipping.

        Raises:
            InvalidClippingRangeError: The clipping range is invalid.
        """

        # Initialize the base class.
        #
        super().__init__()

        # Initialize members.
        #
        self.__classes = 0     # Number of classes. The input labels are expected in the [0, classes-1] range.
        self.__clip_min = 0.0  # Minimum value for clipping.
        self.__clip_max = 1.0  # Maximum value for clipping.

        # Configure label map.
        #
        self.__setclasses(classes=classes)
        self.__setclipping(clip_min=clip_min, clip_max=clip_max)

    def __setclasses(self, classes):
        """
        Configure the number of expected classes.

        Args:
            classes (int): Number of classes.
        """

        self.__classes = classes

    def __setclipping(self, clip_min, clip_max):
        """
        Configure the clipping parameters.

        Args:
            clip_min (float): Minimum value for clipping.
            clip_max (float): Maximum value for clipping.

        Raises:
            InvalidClippingRangeError: The clipping range is invalid.
        """

        # Check if the clipping range is valid.
        #
        if clip_max <= clip_min:
            raise dptweighterrors.InvalidClippingRangeError(clip_min, clip_max)

        # Save clipping parameters.
        #
        self.__clip_min = clip_min
        self.__clip_max = clip_max

    @property
    def classes(self):
        """
        Get the number of expected classes. The valid labels are [0, classes-1]

        Returns:
            int: Number of classes.
        """

        return self.__classes

    @property
    def range(self):
        """
        Get the clipping range.

        Returns:
            tuple: Clipping range.
        """

        return self.__clip_min, self.__clip_max

    def _weigh(self, patches):
        """
        Calculate weight map.

        Args:
            patches (dict): A {level: {'patches': patches, 'labels': labels}} dictionary with extracted patches, corresponding labels or label patches.

        Returns:
            dict: Dictionary with added weight maps for each level.
        """

        return patches

    def calculate(self, patches):
        """
        Calculate weight map and remove the valid pixel map from the collection.

        Args:
            patches (dict): A {level: {'patches': patches, 'labels': labels}} dictionary with extracted patches, corresponding labels or label patches.

        Returns:
            dict: Dictionary with added weight maps for each level.
        """

        # Add the weight map.
        #
        patches = self._weigh(patches=patches)

        # Delete the valid pixel map, since it is not necessary any more.
        #
        for level in patches:
            if 'valid' in patches[level]:
                del patches[level]['valid']

        return patches
