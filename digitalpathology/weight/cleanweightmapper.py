"""
This file contains a class that creates weight maps for U-Net like networks.
"""

from . import weightmapperbase as dptweightmapperbase

import numpy as np

#----------------------------------------------------------------------------------------------------

class CleanWeightMapper(dptweightmapperbase.WeightMapperBase):
    """This class creates weight map that is 1.0 where there are valid labels and 0.0 elsewhere."""

    def __init__(self, classes):
        """
        Initialize the object.

        Args:
            classes (int): Number of classes. The input labels are expected in the [0, classes-1] range.
        """

        # Initialize the base class.
        #
        super().__init__(classes=classes, clip_min=0.0, clip_max=1.0)

    def _weigh(self, patches):
        """
        Calculate weight map.

        Args:
            patches (dict): A {level: {'patches': patches, 'labels': labels, 'valid': valid_map}} dictionary with extracted patches, corresponding labels or label patches.

        Returns:
            dict: Dictionary with added weight maps for each level.
        """

        # Insert new level in the normalized batch dictionary using the valid pixel map.
        #
        for level in patches:
            patches[level]['weights'] = patches[level]['valid'].astype(dtype=np.float32)

        return patches
