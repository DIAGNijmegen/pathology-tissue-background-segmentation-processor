"""
This file contains class normalizing the image patches that were extracted from whole-slide image to a target range.
"""

from . import normalizerbase as dptnormalizerbase

from ..errors import normalizationerrors as dptnormalizationerrors
import numpy as np

#----------------------------------------------------------------------------------------------------

class RgbNormalizer(dptnormalizerbase.NormalizerBase):
    """
    This class can normalize a batch of RGB image patches with [0, 255] value range and labels that were extracted from whole-slide images to a target value range.
    The data type of the image will be np.float32 anf the data type of th labels will be np.int32 after normalization.
    """

    def __init__(self, target_range):
        """
        Initialize the object.

        Args:
            target_range (tuple): Target range. Tuple of two.

        Raises:
            InvalidNormalizationRangeError: Target range is invalid.
        """

        # Initialize the base class.
        #
        super().__init__()

        # Initialize members.
        #
        self.__target_range = None      # Target range of normalization.
        self.__target_range_size = 0.0  # Calculated size of the target range.

        # Process the configured parameters.
        #
        self.__setrange(target_range)

    def __setrange(self, target_range):
        """
        Configure the target range.

        Args:
            target_range (tuple): Target range. Tuple of two.

        Raises:
            InvalidNormalizationRangeError: The target range is invalid.
        """

        # Check target range.
        #
        if len(target_range) < 2 or target_range[1] <= target_range[0]:
            dptnormalizationerrors.InvalidNormalizationRangeError('target', target_range)

        # Save the range.
        #
        self.__target_range = (float(target_range[0]), float(target_range[1]))
        self.__target_range_size = float(target_range[1] - target_range[0])

    def normalize(self, batch):
        """
        Normalize the batch to the target range.

        Args:
            batch (dict): A {level: {'patches': patches, 'labels': labels, 'weights': weights}} dictionary with extracted patches, corresponding labels or label patches and weight maps.

        Returns:
            dict: Batch dictionary with normalized image patches and labels.
        """

        normalized_batch = {}
        for level in batch:
            # Insert new level in the normalized batch dictionary.
            #
            normalized_batch[level] = {}

            # Convert the batch to float representation to the target interval.
            #
            normalized_batch[level]['patches'] = batch[level]['patches'].astype(np.float32) * (self.__target_range_size / 255.0) + self.__target_range[0]

            # Reformat labels to np.int32 data type.
            #
            normalized_batch[level]['labels'] = batch[level]['labels'].astype(np.int32)

            # Add the remaining data.
            #
            if 'weights' in batch[level]:
                normalized_batch[level]['weights'] = batch[level]['weights']

        return normalized_batch
