"""
This file contains a base class for calculating weight maps that normalizes based on pixel counts in a set of image patches for U-Net like networks.
"""

from .. import weightmapperbase as dptweightmapperbase

import numpy as np

#----------------------------------------------------------------------------------------------------

class NormalizingWeightMapperBases(dptweightmapperbase.WeightMapperBase):
    """Base class for weight map calculation that normalizes over a set of patches. It ensures that all the labels have equal weights in the patch."""

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
        super().__init__(classes=classes, clip_min=clip_min, clip_max=clip_max)

        # Initialize members.
        #
        self.__normalize = normalize

    @property
    def normalize(self):
        """
        Get the normalization flag.

        Returns:
            bool: True of the weight mapper normalizes on patch size, false if normalizes on valid pixel count.
        """

        return self.__normalize

    def _weightsofpatch(self, labels, valid):
        """
        Calculate weight map in the patches. It checks how often a label is represented in the patches and calculates a ratio to ensure that all the labels have equal weights in the patches.

        Args:
            labels (np.ndarray): Patches or a single patch of labels.
            valid (np.ndarray): Patches or a single patch of valid pixels.

        Returns:
            np.ndarray: Calculated weight for the patches.
        """

        # Count labels in label patches, the number of valid pixels, and how many pixels should belong to a given label if they were balanced.
        #
        label_counts = np.bincount(labels.flatten(), weights=valid.flatten(), minlength=self.classes)
        valid_pixel_count = np.sum(valid, axis=None)
        balanced_count = valid_pixel_count / np.sum(np.clip(label_counts, a_min=0.0, a_max=1.0), axis=None)

        # Calculate the final weight per network label.
        #
        network_label_weights = np.zeros(shape=label_counts.shape, dtype=np.float32)
        network_label_weights = np.divide(np.full(shape=label_counts.shape, fill_value=balanced_count, dtype=label_counts.dtype), label_counts, out=network_label_weights, where=0.0 < label_counts)
        network_label_weights = np.clip(network_label_weights, a_min=self.range[0], a_max=self.range[1])

        # Assign the calculated weights to each label in the patch.
        #
        weights = network_label_weights[labels]
        weights[np.logical_not(valid)] = 0.0

        # Normalize the weight values.
        #
        normalization_ratio = (labels.size if self.__normalize else valid_pixel_count) / np.sum(weights, axis=None)
        weights *= normalization_ratio

        # Return the calculated weights.
        #
        return weights
