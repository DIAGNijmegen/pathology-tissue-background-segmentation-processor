"""
This file contains a class that can map image label values to network label values..
"""

import digitalpathology.errors.labelerrors as dptlabelerrors

import numpy as np

#----------------------------------------------------------------------------------------------------

class LabelMapper(object):
    """Label value mapper class: maps the image labels to network labels."""

    def __init__(self, label_map):
        """
        Initialize the object.

        Args:
            label_map (dict): Dictionary mapping image labels to network labels. All labels in the mask should be mapped.

        Raises:
            EmptyLabelMapError: The label value to label index map is empty.
            InvalidKeyLabelError: Invalid key in the label map.
            NonContinuousLabelListError: Invalid value in the label map.
        """

        # Initialize the base class.
        #
        super().__init__()

        # Initialize members.
        #
        self.__label_map = {}           # Image label to network label mapping.
        self.__image_labels = []        # List of known image labels.
        self.__label_lookup = None      # Network value lookup table.
        self.__mapped_to_zero = 0       # Image label value that is mapped to the zero network label.
        self.__network_label_count = 0  # Number of target (network) labels.

        # Configure the label mapping.
        #
        self.__setlabelmap(label_map)

    def __setlabelmap(self, label_map):
        """
        Store the label map.

        Args:
            label_map (dict): Label value to index mapping.

        Raises:
            EmptyLabelMapError: The label value to label index map is empty.
            InvalidKeyLabelError: Invalid key in the label map.
            NonContinuousLabelListError: Invalid value in the label map.
        """

        # Check labels.
        #
        if not label_map:
            raise dptlabelerrors.EmptyLabelMapError()

        # Check if all image labels are valid.
        #
        if min(label_map.keys()) < 0 or np.iinfo(np.uint8).max <= max(label_map.keys()):
            raise dptlabelerrors.InvalidKeyLabelError(label_map)

        # Check if the target (network) labels are a continuous interval starting with 0.
        #
        if set(label_map.values()) != set(range(max(label_map.values())+1)):
            raise dptlabelerrors.NonContinuousLabelListError(label_map)

        # Initialize the label map.
        #
        self.__label_lookup = np.zeros(shape=max(label_map.keys()) + 1, dtype=np.int)

        # Fill in the mapping and find the image label that is mapped to zero.
        #
        for image_label in label_map:
            self.__label_lookup[image_label] = label_map[image_label]

            if label_map[image_label] == 0:
                self.__mapped_to_zero = image_label

        # Store the original label mapping and count the number of labels.
        #
        self.__label_map = label_map
        self.__image_labels = list(label_map.keys())
        self.__network_label_count = len(set(label_map.values()))

    @property
    def mapping(self):
        """
        Get image label to network label dictionary.

        Returns:
            dict: get the original label map.
        """

        return self.__label_map

    @property
    def classes(self):
        """
        Get the number of classes after mapping.

        Returns:
            int: Number of classes after mapping.
        """

        return self.__network_label_count

    def process(self, patches, valid_map=False):
        """
        Map all label values to indices in the patch collection.

        Args:
            patches (dict): dictionary {level: {'patches': patch array, 'labels': label array} with the patch and label arrays. These arrays are both np.array types.
            valid_map (bool): Flag to control if valid label map should be also generated.

        Returns:
            dict: Processed patch collection with a boolean array at the 'valid' key that maps the valid labels.
        """

        # Apply replacement by removing all invalid labels from the image and mapping the valid labels.
        #
        for level_data in patches.values():
            if valid_map:
                level_data['valid'] = np.isin(element=level_data['labels'], test_elements=self.__image_labels)
                level_data['labels'][np.logical_not(level_data['valid'])] = self.__mapped_to_zero

            level_data['labels'] = self.__label_lookup[level_data['labels']]

        return patches
