"""
This file contains a base class for calculating weight maps that normalizes based on pixel counts in a batch of patches for U-Net like networks.
"""

from . import normalizingweightmapperbase as dptnormalizingweightmapperbase

#----------------------------------------------------------------------------------------------------

class BatchWeightMapper(dptnormalizingweightmapperbase.NormalizingWeightMapperBases):
    """Class for weight map calculation per batch. It ensures that all the labels have equal weights in the batch."""

    def __init__(self, classes, normalize, clip_min, clip_max):
        """
        Initialize the object.

        Args:
            classes (int): Number of classes. The input labels are expected in the [0, classes-1] range.
            normalize (bool): Normalize on patch size.
            clip_min (float): Minimum value for clipping.
            clip_max (float): Maximum value for clipping.

        Raises:
            InvalidClippingRangeError: The clipping range is invalid.
        """

        # Initialize the base class.
        #
        super().__init__(classes=classes, normalize=normalize, clip_min=clip_min, clip_max=clip_max)

    def _weigh(self, patches):
        """
        Calculate weight map in a patch. It checks how often a label is represented in a patch and calculates a ratio to ensure that all the labels have equal weights in a batch.

        Args:
            patches (dict): A {level: {'patches': patches, 'labels': labels, 'weights': weights}} dictionary with extracted patches, corresponding labels or label patches and weight maps.

        Returns:
            dict: Batch dictionary with weight maps.
        """

        # Calculate for each patch at each level.
        #
        for level in patches:
            patches[level]['weights'] = self._weightsofpatch(labels=patches[level]['labels'], valid=patches[level]['weights'])

        return patches
