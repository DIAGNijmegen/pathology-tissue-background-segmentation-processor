"""
This file contains class for organizing the data source list into per-purpose/per-category lists.
"""

from ..errors import dataerrors as dptdataerrors
from ..errors import configerrors as dptconfigerrors
from ..patch import patchsource as dptpatchsource
from ..utils import population as dptpopulation

import json
import yaml
import random
import os
import math

#----------------------------------------------------------------------------------------------------

class BatchSource(object):
    """This class is a batch source class that distribute the data source list into per-purpose/per-category lists."""

    def __init__(self, source_items=None):
        """
        Initialize the object.

        Args:
            source_items (dict, None): Category ID -> list of PatchSource or dictionary source items map. Where the dictionaries have 'image', 'mask', 'stat' and 'labels' keys.
        """

        # Initialize the base class.
        #
        super().__init__()

        # Initialize members.
        #
        self.__source_items = []          # Patch source items.
        self.__categories = {}            # Patch source indices for each category.
        self.__purposes = {}              # Patch source indices for each purpose.
        self.__purpose_distribution = {}  # Purpose image distribution.
        self.__path_replacements = {}     # Path replacements.

        # Save parameters and read data from JSON or YAML.
        #
        self.__pushcategories(source_items=source_items)  # Save source items.

    def __pushsourceitems(self, category_id, item_list):
        """
        Add list of source items to a category.

        Args:
            category_id (str): Category identifier.
            item_list (list): List of source items (from file) to add. The items can be PatchSource objects or dictionaries with 'image', 'mask', 'stat' and 'labels' keys.

        Returns:
            set: Indices that have been added to the category.
        """

        # Create an empty index set for the category.
        #
        if category_id not in self.__categories:
            self.__categories[category_id] = set()

        # Create instances from the list and add it to the category.
        #
        item_indices = set()
        for source_item in item_list:
            if type(source_item) == dptpatchsource.PatchSource:
                # Load source item from a PatchSource object.
                #
                self.__source_items.append(dptpatchsource.PatchSource(image_path=source_item.image,
                                                                      mask_path=source_item.mask,
                                                                      stat_path=source_item.stat,
                                                                      available_labels=source_item.labels))
            else:
                # Load source item from a dictionary with 'image', 'mask', 'stat' and 'labels' keys (where 'mask' or 'stat' is optional).
                #
                self.__source_items.append(dptpatchsource.PatchSource(image_path=source_item['image'],
                                                                      mask_path=source_item.get('mask', ''),
                                                                      stat_path=source_item.get('stat', ''),
                                                                      available_labels=tuple(source_item['labels'])))

            item_indices.add(len(self.__source_items) - 1)

        # Add the items to the category.
        #
        self.__categories[category_id].update(item_indices)

        return item_indices

    def __pushcategories(self, source_items):
        """
        Save the per category source item list.

        Args:
            source_items (dict, None): Category ID -> list of PatchSource source items map.

        Returns:
            set: Indices that have been added.
        """

        # Save all category lists.
        #
        item_indices = set()
        if source_items:
            for category_id in source_items:
                added_indices = self.__pushsourceitems(category_id=category_id, item_list=source_items[category_id])
                item_indices.update(added_indices)

        return item_indices

    def __sourceitem(self, item_index, apply_replacements):
        """
        Get a copy of a source item.

        Args:
            item_index (int): Index of the source item.
            apply_replacements (bool): Apply path replacements.

        Returns:
            dptpatchsource.PatchSource: patch source item.
        """

        if apply_replacements:
            # Construct normalized paths.
            #
            norm_image_path = os.path.normpath(self.__source_items[item_index].image.format(**self.__path_replacements))
            norm_mask_path = os.path.normpath(self.__source_items[item_index].mask.format(**self.__path_replacements)) if self.__source_items[item_index].mask else ''
            norm_stat_path = os.path.normpath(self.__source_items[item_index].stat.format(**self.__path_replacements)) if self.__source_items[item_index].stat else ''

            return dptpatchsource.PatchSource(image_path=norm_image_path,
                                              mask_path=norm_mask_path,
                                              stat_path=norm_stat_path,
                                              available_labels=self.__source_items[item_index].labels)
        else:
            return dptpatchsource.PatchSource(image_path=self.__source_items[item_index].image,
                                              mask_path=self.__source_items[item_index].mask,
                                              stat_path=self.__source_items[item_index].stat,
                                              available_labels=self.__source_items[item_index].labels)

    def __setpurposedistribution(self, purpose_distribution):
        """
        Save the purpose distribution.

        Args:
            purpose_distribution (dict, None): Data set size ratios for different purposes.

        Raises:
            PurposeListAndRatioMismatchError: The configured purpose distribution does not match the available purposes.
        """

        # Check if the distribution is valid for the available purposes.
        #
        if purpose_distribution.keys() != self.__purposes.keys():
            dptconfigerrors.PurposeListAndRatioMismatchError(purpose_distribution, list(self.__purposes.keys()))

        # Save the distribution.
        #
        self.__purpose_distribution = purpose_distribution

    def __collect(self, purpose_id=None, category_id=None):
        """
        Get the set of source item indices for the given purposes in the given categories.

        Args:
            purpose_id (str, list, None): Purpose identifiers. All purposes if None.
            category_id (str, list, None): Category identifiers. All categories if None.

        Returns:
            set: Set if source item indices.
        """

        # Filter the data set by selection.
        #
        if purpose_id is None and category_id is None:
            # The number of all items.
            #
            return set(range(len(self.__source_items)))

        elif purpose_id is None and category_id is not None:
            # The number of items in the given categories.
            #
            category_id_list = category_id if type(category_id) == list or type(category_id) == tuple else [category_id]
            return set.union(*[self.__categories.get(category_id, set()) for category_id in category_id_list])

        elif purpose_id is not None and category_id is None:
            # The number of items in the given purposes.
            #
            purpose_id_list = purpose_id if type(purpose_id) == list or type(purpose_id) == tuple else [purpose_id]
            return set.union(*[self.__purposes.get(purpose_id, set()) for purpose_id in purpose_id_list])

        else:
            # Filter both purposes and categories.
            #
            category_id_list = category_id if type(category_id) == list or type(category_id) == tuple else [category_id]
            purpose_id_list = purpose_id if type(purpose_id) == list or type(purpose_id) == tuple else [purpose_id]

            return set.intersection(set.union(*[self.__categories.get(category_id, set()) for category_id in category_id_list]),
                                    set.union(*[self.__purposes.get(purpose_id, set()) for purpose_id in purpose_id_list]))

    def purposes(self):
        """
        Get the list of configured purposes.

        Returns:
            list: Available purposes.
        """

        return list(self.__purposes.keys())

    def categories(self, purpose_id=None):
        """
        Get a set of the configured categories for the given purposes or all purposes.

        Args:
            purpose_id (str, list, None): Purpose identifiers. All purposes if None.

        Returns:
            list: Available categories.
        """

        if purpose_id is None:
            # Return all available categories for all purposes.
            #
            return list(self.__categories.keys())
        else:
            # Filter the categories by purposes.
            #
            purpose_indices = self.__collect(purpose_id=purpose_id, category_id=None)

            categories = set()
            for index in purpose_indices:
                for category_id in self.__categories:
                    if index in self.__categories[category_id]:
                        categories.add(category_id)

            return sorted(list(categories))

    def distribution(self):
        """
        Get the purpose distribution ratio map.

        Returns:
            dict: Purpose distribution.
        """

        return self.__purpose_distribution.copy()

    def replacements(self):
        """
        Get the path replacements.

        Returns:
            dict: Path replacement dictionary.
        """

        return self.__path_replacements.copy()

    def update(self, path_replacements):
        """
        Update the path replacement dictionary.

        Args:
            path_replacements (dict): Path replacements dictionary.
        """

        self.__path_replacements.update(path_replacements)

    def distribute(self, purpose_distribution):
        """
        Distribute the source items into purposes.

        Args:
            purpose_distribution (dict, None): Data set size ratios for different purposes. If empty or None the current purposes deleted.
        """

        # Check the purpose distribution. For empty dictionary all purposes are deleted.
        #
        if purpose_distribution:
            # Set the new distribution and reset the purposes.
            #
            self.__setpurposedistribution(purpose_distribution=purpose_distribution)
            self.__purposes = {purpose_id: set() for purpose_id in purpose_distribution}

            # Normalize purpose ratios to sum of 1.0.
            #
            ratio_sum = sum(purpose_distribution.values())
            normalized_purpose_distribution = {purpose_id: purpose_distribution[purpose_id] / ratio_sum for purpose_id in purpose_distribution}

            # Distribute the per-category lists into the purpose groups.
            #
            for category_id in self.__categories:
                category_population = len(self.__categories[category_id])
                minimum_count = 1 if len(normalized_purpose_distribution) <= category_population else 0
                category_purpose_ratios = {purpose_id: (purpose_ratio, minimum_count, category_population) for purpose_id, purpose_ratio in normalized_purpose_distribution.items()}
                category_purpose_distributed = dptpopulation.distribute_population(population=category_population, ratios=category_purpose_ratios)

                category_items = list(self.__categories[category_id])
                random.shuffle(category_items)

                count_sum = 0
                for purpose_id in category_purpose_distributed:
                    self.__purposes[purpose_id].update(category_items[count_sum: count_sum + category_purpose_distributed[purpose_id]])
                    count_sum += category_purpose_distributed[purpose_id]
        else:
            # Delete the current purposes.
            #
            self.__purposes = {}
            self.__purpose_distribution = {}

    def validate(self, purpose_distribution):
        """
        Check if the given purpose distribution is the same as the configured one.

        Args:
            purpose_distribution (dict): Data set size ratios for different purposes. If empty or None the current purposes deleted.

        Returns:
            bool: True if they are the same, false otherwise.
        """

        if self.__purpose_distribution:
            if all(purpose in self.__purpose_distribution for purpose in purpose_distribution):
                # Normalize purpose ratios to sum of 1.0.
                #
                ratio_sum = sum(purpose_distribution.values())
                normalized_purpose_distribution = {purpose_id: purpose_distribution[purpose_id] / ratio_sum for purpose_id in purpose_distribution}

                current_ratio_sum = sum(self.__purpose_distribution[purpose] for purpose in purpose_distribution)
                normalized_current_purpose_distribution = {purpose_id: self.__purpose_distribution[purpose_id] / current_ratio_sum for purpose_id in purpose_distribution}

                # Check if the ratios are the same.
                #
                return all(math.isclose(normalized_purpose_distribution[purpose_id], normalized_current_purpose_distribution[purpose_id]) for purpose_id in purpose_distribution)
            else:
                # The list of purposes cannot be matched.
                #
                return False
        else:
            # Cannot validate against invalid distribution.
            #
            return False

    def push(self, source_items, purpose_id=None):
        """
        Add items.

        Args:
            source_items (dict): Category ID -> list of PatchSource or dictionary source items map. Where the dictionaries have 'image', 'mask', 'stat' and 'labels' keys.
            purpose_id (str, None): Purpose identifier.
        """

        # Push new items.
        #
        item_indices = self.__pushcategories(source_items=source_items)

        # Check if the items have been added to a specific purpose.
        #
        if purpose_id is not None:
            # Add the items to the purpose.
            #
            if purpose_id not in self.__purposes:
                self.__purposes[purpose_id] = set()

            self.__purposes[purpose_id].update(item_indices)

            # Recalculate the purpose distribution.
            #
            self.__purpose_distribution = {purpose_id: float(len(self.__purposes[purpose_id])) for purpose_id in self.__purposes}

    def count(self, purpose_id=None, category_id=None):
        """
        Get the number of source items for the given purposes in the given categories.

        Args:
            purpose_id (str, list, None): Purpose identifiers. All purposes if None.
            category_id (str, list, None): Category identifiers. All categories if None.

        Returns:
            int: Number of source items for the given purposes in the given categories.
        """

        return len(self.__collect(purpose_id=purpose_id, category_id=category_id))

    def items(self, purpose_id=None, category_id=None, replace=True):
        """
        Get the set of source items for the given purposes in the given categories.

        Args:
            purpose_id (str, list, None): Purpose identifiers. All purposes if None.
            category_id (str, list, None): Category identifiers. All categories if None.
            replace (bool): Apply path replacements.

        Returns:
            set: Set of source items for the given purposes in the given categories.
        """

        # Collect the indices.
        #
        indices = self.__collect(purpose_id=purpose_id, category_id=category_id)

        # Build output.
        #
        return set([self.__sourceitem(item_index=index, apply_replacements=replace) for index in indices])

    def collection(self, purpose_id=None, category_id=None, replace=True):
        """
        Get the set of source items for the given purposes in the given categories organized into per category sets.

        Args:
            purpose_id (str, list, None): Purpose identifiers. All purposes if None.
            category_id (str, list, None): Category identifiers. All categories if None.
            replace (bool): Apply path replacements.

        Returns:
            dict: Dictionary mapping categories to set of PatchSource items.
        """

        # Assemble the list of category IDs to query.
        #
        category_id_list = category_id if type(category_id) == list or type(category_id) == tuple else [category_id]
        category_id_query = self.__categories if category_id is None else category_id_list

        return {category_id_item: self.items(purpose_id=purpose_id, category_id=category_id_item, replace=replace) for category_id_item in category_id_query}

    def load(self, file_path):
        """
        Load category id, image path, mask path triplets from JSON or YAML file. The JSON/YAML file can be of two different types. If the 'type' value in the JSON/YAML file is
        'list' then the content of the file is just a per-category list that is not distributed per-purpose. If the 'type' value is 'distributed' than the content of the file
        is already distributed per-purpose.

        Args:
            file_path (str): Data file path.

        Returns:
            dict: dictionary of image category id-s. Each image category id points to a list of maps with 'image', 'mask', 'stat' and 'labels' key values.

        Raises:
            InvalidDataFileExtensionError: The format cannot be derived from the file extension.
            InvalidDataSourceTypeError: Invalid JSON or YAML file.
            PurposeListAndRatioMismatchError: The configured purpose distribution does not match the available purposes.
        """

        # Check file extension.
        #
        dump_format = os.path.splitext(file_path)[1].lower()
        if dump_format not in ['.json', '.yaml']:
            raise dptconfigerrors.InvalidDataFileExtensionError(file_path)

        # Load data structure.
        #
        data_content = {}
        with open(file=file_path, mode='r') as data_file:
            if dump_format == '.json':
                data_content = json.load(fp=data_file)
            elif dump_format == '.yaml':
                data_content = yaml.load(stream=data_file, Loader=yaml.SafeLoader)

        # Convert the source format to the internal representation.
        #
        if data_content['type'] == 'list':
            # Process all categories.
            #
            for category_id in data_content['data']:
                self.__pushsourceitems(category_id=category_id, item_list=data_content['data'][category_id])
        elif data_content['type'] == 'distributed':
            # Process all purposes.
            #
            for purpose_id in data_content['data']:
                if purpose_id not in self.__purposes:
                    self.__purposes[purpose_id] = set()

                # Process all categories in the purpose.
                #
                for category_id in data_content['data'][purpose_id]:
                    added_indices = self.__pushsourceitems(category_id=category_id, item_list=data_content['data'][purpose_id][category_id])

                    self.__purposes[purpose_id].update(added_indices)

            # Save the purpose distribution.
            #
            self.__setpurposedistribution(purpose_distribution=data_content['distribution'])
        else:
            # JSON or YAML type identifier is unknown.
            #
            raise dptdataerrors.InvalidDataSourceTypeError(data_content['type'])

        # Save the path replacements.
        #
        self.update(path_replacements=data_content.get('path', {}))

    def save(self, file_path):
        """
        Save the data to a JSON or a YAML file.

        Args:
            file_path (str): Target file path. The file format is derived from the file extension.

        Raises:
            InvalidDataFileExtensionError: The format cannot be derived from the file extension.
        """

        # Save internal data to json or yaml.
        #
        with open(file=file_path, mode='w') as data_file:
            # Convert data to serializable format.
            #
            serializable_data = {'path': self.__path_replacements, 'data': {}}

            if self.__purpose_distribution:
                serializable_data['type'] = 'distributed'
                serializable_data['distribution'] = self.__purpose_distribution

                # Save all purposes.
                #
                for purpose_id in self.__purposes:
                    purpose_serializable_data = {}

                    # Save all categories.
                    #
                    for category_id in self.__categories:
                        # Collect the item indices for this category.
                        #
                        category_indices = self.__collect(purpose_id=purpose_id, category_id=category_id)
                        if category_indices:
                            purpose_serializable_data[category_id] = []

                            # Convert PatchSource objects to serializable dictionaries.
                            #
                            for item_index in category_indices:
                                serializable_patch_source = {'image': self.__source_items[item_index].image}
                                if self.__source_items[item_index].mask:
                                    serializable_patch_source['mask'] = self.__source_items[item_index].mask
                                if self.__source_items[item_index].stat:
                                    serializable_patch_source['stat'] = self.__source_items[item_index].stat
                                serializable_patch_source['labels'] = self.__source_items[item_index].labels

                                purpose_serializable_data[category_id].append(serializable_patch_source)

                    # Save the purpose data.
                    #
                    if purpose_serializable_data:
                        serializable_data['data'][purpose_id] = purpose_serializable_data
            else:
                serializable_data['type'] = 'list'

                # Save all categories.
                #
                for category_id in self.__categories:
                    # Collect the item indices for this category.
                    #
                    category_indices = self.__collect(purpose_id=None, category_id=category_id)
                    if category_indices:
                        serializable_data['data'][category_id] = []

                        # Convert PatchSource objects to serializable dictionaries.
                        #
                        for item_index in category_indices:
                            serializable_patch_source = {'image': self.__source_items[item_index].image}
                            if self.__source_items[item_index].mask:
                                serializable_patch_source['mask'] = self.__source_items[item_index].mask
                            if self.__source_items[item_index].stat:
                                serializable_patch_source['stat'] = self.__source_items[item_index].stat
                            serializable_patch_source['labels'] = self.__source_items[item_index].labels

                            serializable_data['data'][category_id].append(serializable_patch_source)

            # Save data to json or yaml format.
            #
            file_format = os.path.splitext(file_path)[1].lower()
            if file_format == '.json':
                json.dump(serializable_data, fp=data_file, indent=4)
            elif file_format == '.yaml':
                yaml.dump(data=serializable_data, stream=data_file, indent=4, default_flow_style=False)
            else:
                # Unknown file format.
                #
                raise dptconfigerrors.InvalidDataFileExtensionError(file_path)
