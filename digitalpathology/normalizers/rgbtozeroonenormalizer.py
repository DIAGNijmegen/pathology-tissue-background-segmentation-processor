"""
This file contains class normalizing the image patches that were extracted from whole-slide image to a target range.
"""

from . import normalizerbase as dptnormalizerbase

import numpy as np

#----------------------------------------------------------------------------------------------------

class RgbToZeroOneNormalizer(dptnormalizerbase.NormalizerBase):
    """
    This class can normalize a batch of RGB image patches with [0, 255] value range and labels that were extracted from whole-slide images to [0.0, 1.0] target value range.
    The data type of the image will be np.float32 anf the data type of th labels will be np.int32 after normalization.
    """

    def __init__(self):
        """Initialize the object."""

        # Initialize the base class.
        #
        super().__init__()

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
            normalized_batch[level]['patches'] = batch[level]['patches'].astype(np.float32) / 255.0

            # Reformat labels to np.int32 data type.
            #
            normalized_batch[level]['labels'] = batch[level]['labels'].astype(np.int32)

            # Add the remaining data.
            #
            if 'weights' in batch[level]:
                normalized_batch[level]['weights'] = batch[level]['weights']

        return normalized_batch
