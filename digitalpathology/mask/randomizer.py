"""
This file contains class for random selecting indexes from mr-image.
"""

from ..errors import labelerrors as dptlabelerrors

import numpy as np

#----------------------------------------------------------------------------------------------------

class IndexRandomizer(object):
    """Randomly select indices from mr-image."""

    def __init__(self, pixel_counts):
        """
        Initialize the object and configure the operation mode with labels and their numbers. The count list should not be empty and it should
        contain only non-negative values with at least one positive.

        Args:
            pixel_counts (dict): Label value to pixel count map.

        Raises:
            EmptyLabelListError: The list of labels is empty.
            NegativeLabeledItemCountError: The label to count list contains negative count value.
            AllZeroLabelCountsError: The values in the count list are all zeros.
        """

        # Initialize the base class.
        #
        super().__init__()

        # Initialize source.
        #
        self.__pixel_counts = {}  # Per-label pixel count list

        # Configure the operation mode.
        #
        self.__setcounts(pixel_counts)

    def __setcounts(self, pixel_counts):
        """
        Configure the labels and their numbers.

        Args:
            pixel_counts (dict): Label value to pixel count map.

        Raises:
            EmptyLabelListError: The list of labels is empty.
            NegativeLabeledItemCountError: The label to count list contains negative count value.
            AllZeroLabelCountsError: The values in the count list are all zeros.
        """

        # The label count list should not be empty and contain only non-negative values and at least one positive.
        #
        if not pixel_counts or not len(pixel_counts):
            raise dptlabelerrors.EmptyLabelListError()

        if min(count for count in pixel_counts.values()) < 0:
            raise dptlabelerrors.NegativeLabeledItemCountError(pixel_counts)

        if max(count for count in pixel_counts.values()) <= 0:
            raise dptlabelerrors.AllZeroLabelCountsError()

        # Save the label counts.
        #
        self.__pixel_counts = pixel_counts

    def randomindices(self, counts):
        """
        Randomize the given number of pixel indices with uniform distribution. Each generated index means the n-th pixel in row-continuous order of the given label.

        Args:
            counts (dict): Label value to label count to randomize.

        Returns:
            dict: List of array of indexes per label value.

        Raises:
            LabelListMismatchError: The available labels does not match the requested labels
        """

        if any(label not in self.__pixel_counts for label in counts):
            raise dptlabelerrors.LabelListMismatchError(list(self.__pixel_counts.keys()), list(counts.keys()))

        # Generate random integers.
        #
        return {label: np.random.randint(low=0, high=self.__pixel_counts[label], size=counts[label]) for label in counts}
