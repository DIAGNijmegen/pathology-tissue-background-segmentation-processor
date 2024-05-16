"""
This file contains class normalizing the image patches that were extracted from whole-slide image to a target range.
"""

from . import normalizerbase as dptnormalizerbase

#----------------------------------------------------------------------------------------------------

class PassThroughNormalizer(dptnormalizerbase.NormalizerBase):
    """
    Returns the batch without modification.
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

        return batch
