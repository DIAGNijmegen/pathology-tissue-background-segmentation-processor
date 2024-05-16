"""
This file contains a wrapper class for converting annotations to multi-resolution image masks.
"""

from . import imagereader as dptimagereader
from ..errors import imageerrors as dptimageerrors

import multiresolutionimageinterface as mir

import scipy.ndimage
import skimage.measure
import numpy as np
#import rdp
import os

#----------------------------------------------------------------------------------------------------

class Annotation(object):
    """Wrapper class for ASAP annotations."""

    def __init__(self):
        """Initialize the object."""

        # Initialize the base class.
        #
        super().__init__()

        # Initialize members.
        #
        self.__groups = {}       # Annotation group properties.
        self.__annotations = []  # Annotation items.
        self.__path = None       # Path of the opened annotation.

        self.__default_group_color = (100, 254, 46)                                                                 # Default group color.
        self.__default_annotation_color = (244, 250, 88)                                                            # Default annotation color.
        self.__type_mapping = {'dot': 1, 'polygon': 2, 'spline': 3, 'points': 4, 'measurement': 5, 'rectangle': 6}  # Annotation type mapping.

    def __testgroup(self, group_name):
        """
        Tests if a group can be removed and removes it if possible.

        Args:
            group_name (str): Name of the group to test and remove.
        """

        groups_to_test = {group_name}
        while groups_to_test:
            test_group_name = groups_to_test.pop()

            if test_group_name in self.__groups:
                if not any(test_group_name == annotation_item['group'] for annotation_item in self.__annotations):
                    if not any(test_group_name == self.__groups[group_item]['group'] for group_item in self.__groups):
                        if self.__groups[test_group_name]['group'] is not None:
                            groups_to_test.add(self.__groups[test_group_name]['group'])

                        del self.__groups[test_group_name]

    def __build(self, spacing):
        """
        Build multiresolutionimageinterface type annotation object from the stored data. If set the stored coordinates are understood as spatial coordinates
        and the spacing is used to scale them to pixel coordinates.

        Args:
            spacing (float, None): Spacing.

        Returns:
            mir.AnnotationList: Annotation object.
        """

        # Create an annotation objects.
        #
        mir_annotation = mir.AnnotationList()

        # Construct the groups.
        #
        mir_group_map = {}
        for group_name in self.__groups:
            group_color = self.__groups[group_name]['color']
            mir_group_item = mir.AnnotationGroup()
            mir_group_item.setName(group_name)
            mir_group_item.setColor('#{red:02X}{green:02X}{blue:02X}'.format(red=group_color[0], green=group_color[1], blue=group_color[2]))

            mir_group_map[group_name] = mir_group_item

        for group_name in self.__groups:
            if self.__groups[group_name]['group'] is not None:
                mir_group_map[group_name].setGroup(mir_group_map[self.__groups[group_name]['group']])

        mir_annotation.setGroups(list(mir_group_map.values()))

        # Construct the annotation items.
        #
        mir_annotation_list = []
        for annotation_item in self.__annotations:
            mir_annotation_item = mir.Annotation()
            mir_annotation_item.setName(annotation_item['name'])
            mir_annotation_item.setType(self.__type_mapping[annotation_item['type']])
            mir_annotation_item.setColor('#{red:02X}{green:02X}{blue:02X}'.format(red=annotation_item['color'][0], green=annotation_item['color'][1], blue=annotation_item['color'][2]))

            if spacing is None:
                mir_annotation_item.setCoordinates([mir.Point(coordinate_item[1], coordinate_item[0]) for coordinate_item in annotation_item['coordinates']])
            else:
                mir_annotation_item.setCoordinates([mir.Point(coordinate_item[1] / spacing, coordinate_item[0] / spacing) for coordinate_item in annotation_item['coordinates']])

            if annotation_item['group'] is not None:
                mir_annotation_item.setGroup(mir_group_map[annotation_item['group']])

            mir_annotation_list.append(mir_annotation_item)

        mir_annotation.setAnnotations(mir_annotation_list)

        return mir_annotation

    def open(self, annotation_path, spacing=None):
        """
        Open annotation file. ASAP annotations are pixel coordinates on the lowest level, so they depend on the pixel spacing. The internal representation of the coordinates are in micrometers.

        Args:
            annotation_path (str): Path of the annotation to load.
            spacing (float, None): Pixel spacing (micrometers).

        Raises:
            AnnotationOpenError: The specified annotation cannot be opened for reading.
            InvalidFileMissingAnnotationGroupError: The annotation file contains a referenced unknown group.
            InvalidFileInvalidAnnotationCoordinateListError: The list of coordinates is invalid of an annotation.
            InvalidFileUnknownAnnotationTypeError: The annotation file contains an annotation of unknown type.
        """

        # Check if annotation file exists. The loader has a bug an reports non-existent files as successfully opened.
        #
        if not os.path.isfile(annotation_path):
            raise dptimageerrors.AnnotationOpenError(annotation_path)

        # Open the annotation file.
        #
        mir_annotation = mir.AnnotationList()
        mir_repository = mir.XmlRepository(mir_annotation)
        mir_repository.setSource(annotation_path)
        annotation_loaded = mir_repository.load()

        # Check if annotation is successfully opened.
        #
        if not annotation_loaded:
            raise dptimageerrors.AnnotationOpenError(annotation_path)

        # Load annotation groups.
        #
        mir_group_list = mir_annotation.getGroups()
        for mir_group_item in mir_group_list:
            group_name = mir_group_item.getName()
            mir_group_group = mir_group_item.getGroup()
            mir_group_color = mir_group_item.getColor()

            # Get the name of the group.
            #
            group_group = mir_group_group.getName() if mir_group_group is not None else None

            # Convert the HTML color notation to RGB representation.
            #
            group_color = (int(mir_group_color[1:3], 16), int(mir_group_color[3:5], 16), int(mir_group_color[5:7], 16))

            # Save group.
            #
            self.__groups[group_name] = {'group': group_group, 'color': group_color}

        # Check all the groups if the parent group exits.
        #
        for mir_group_item in self.__groups.values():
            if mir_group_item['group'] is not None and mir_group_item['group'] not in self.__groups:
                raise dptimageerrors.InvalidFileMissingAnnotationGroupError(annotation_path, list(self.__groups.keys()), mir_group_item['group'])

        # Load all the annotations.
        #
        annotation_list = mir_annotation.getAnnotations()
        for annotation_item in annotation_list:
            # Retrieve one annotation object.
            #
            annotation_item_coordinate_list = annotation_item.getCoordinates()

            # Collect properties.
            #
            annotation_name = annotation_item.getName()
            mir_annotation_type = annotation_item.getType()
            mir_annotation_group = annotation_item.getGroup()
            mir_annotation_color = annotation_item.getColor()
            if spacing is not None:
                annotation_coordinates = [(annotation_item_coordinate_item.getY() * spacing,
                                           annotation_item_coordinate_item.getX() * spacing) for annotation_item_coordinate_item in annotation_item_coordinate_list]
            else:
                annotation_coordinates = [(annotation_item_coordinate_item.getY(), annotation_item_coordinate_item.getX()) for annotation_item_coordinate_item in annotation_item_coordinate_list]

            # Convert the annotation type identifier to string representation and check if it has the correct number of coordinates.
            #
            if mir_annotation_type == 1:
                annotation_type = 'dot'
                if len(annotation_coordinates) != 1:
                    raise dptimageerrors.InvalidFileInvalidAnnotationCoordinateListError(len(annotation_coordinates), annotation_type, annotation_path)

            elif mir_annotation_type == 2:
                annotation_type = 'polygon'
                if len(annotation_coordinates) < 3:
                    raise dptimageerrors.InvalidFileInvalidAnnotationCoordinateListError(len(annotation_coordinates), annotation_type, annotation_path)

            elif mir_annotation_type == 3:
                annotation_type = 'spline'
                if len(annotation_coordinates) < 3:
                    raise dptimageerrors.InvalidFileInvalidAnnotationCoordinateListError(len(annotation_coordinates), annotation_type, annotation_path)

            elif mir_annotation_type == 4:
                annotation_type = 'points'
                if len(annotation_coordinates) == 0:
                    raise dptimageerrors.InvalidFileInvalidAnnotationCoordinateListError(len(annotation_coordinates), annotation_type, annotation_path)

            elif mir_annotation_type == 5:
                annotation_type = 'measurement'
                if len(annotation_coordinates) != 2:
                    raise dptimageerrors.InvalidFileInvalidAnnotationCoordinateListError(len(annotation_coordinates), annotation_type, annotation_path)

            elif mir_annotation_type == 6:
                annotation_type = 'rectangle'
                if len(annotation_coordinates) != 4:
                    raise dptimageerrors.InvalidFileInvalidAnnotationCoordinateListError(len(annotation_coordinates), annotation_type, annotation_path)

            else:
                raise dptimageerrors.InvalidFileUnknownAnnotationTypeError(annotation_path, mir_annotation_type)

            # Get the name of the group.
            #
            annotation_group = mir_annotation_group.getName() if mir_annotation_group is not None else None

            if annotation_group is not None and annotation_group not in self.__groups:
                raise dptimageerrors.InvalidFileMissingAnnotationGroupError(annotation_path, list(self.__groups.keys()), annotation_group)

            # Convert the HTML color notation to RGB representation.
            #
            annotation_color = (int(mir_annotation_color[1:3], 16), int(mir_annotation_color[3:5], 16), int(mir_annotation_color[5:7], 16))

            # Add the annotation to its group.
            #
            self.__annotations.append({'name': annotation_name, 'type': annotation_type, 'group': annotation_group, 'color': annotation_color, 'coordinates': annotation_coordinates})

        # Save the path.
        #
        self.__path = annotation_path

    def save(self, annotation_path, spacing=None):
        """
        Save the annotation structure to ASAP XML annotation file.

        Args:
            annotation_path (str): Path of the annotation file to save.
            spacing (float, None): Pixel spacing (micrometers).

        Raises:
            InvalidAnnotationPathError: The target annotation path is invalid.
        """

        # Check the target path.
        #
        if not annotation_path:
            raise dptimageerrors.InvalidAnnotationPathError(annotation_path)

        # Construct the annotation object.
        #
        mir_annotation = self.__build(spacing=spacing)

        # Save the annotation to file.
        #
        repository = mir.XmlRepository(mir_annotation)
        repository.setSource(annotation_path)
        repository.save()

    def convert(self, image_path, shape, spacing, label_map=None, conversion_order=None):
        """
        Convert the annotations to a mask multi-resolution image file with the given shape pixel spacing.

        Args:
            image_path (str): Target mask image file path.
            shape (tuple): Required shape of the image.
            spacing (float): Pixel spacing of the mask (micrometer).
            label_map (dict): Mapping of the annotation groups to label values. The values must be in the [0, 255] range.
            conversion_order (list): Annotation group conversion order.

        Raises:
            InvalidMaskImagePathError: The target mask image path is not valid.
            InvalidImageShapeError: Invalid image shape configuration.
            InvalidPixelSpacingError: Invalid pixel spacing configuration.
            LabelMapConversionOrderMismatchError: The label map does not match the conversion order list.
        """

        # Check the target mask image file path.
        #
        if not image_path:
            raise dptimageerrors.InvalidMaskImagePathError(image_path)

        # Check the shape and the spacing.
        #
        if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
            raise dptimageerrors.InvalidImageShapeError(self.__path, shape)

        if spacing is not None and spacing <= 0.0:
            raise dptimageerrors.InvalidPixelSpacingError(self.__path, spacing)

        # Calculate output settings.
        #
        output_dimensions = tuple(reversed(shape))
        output_spacing = (spacing, spacing)
        conversion_label_map = dict(label_map) if label_map is not None else {group_name: 1 for group_name in self.__groups}
        conversion_order_list = list(conversion_order) if conversion_order is not None else list(conversion_label_map.keys())

        # Check the conversion order list against the label map.
        #
        if set(conversion_label_map.keys()) != set(conversion_order_list):
            raise dptimageerrors.LabelMapConversionOrderMismatchError(self.__path, list(label_map.keys()), conversion_order)

        # Construct the annotation object.
        #
        mir_annotation = self.__build(spacing=spacing)

        # Convert the annotation to mask image.
        #
        mir_annotation_mask = mir.AnnotationToMask()
        mir_annotation_mask.convert(mir_annotation, image_path, output_dimensions, output_spacing, conversion_label_map, conversion_order_list)

    def outline(self, image, spacing=None, spacing_tolerance=0.25, label_map=None, single_points=False, rdp_epsilon=1.0):
        """
        Generate annotations by outlining the regions in the mask file.

        The calculation is memory computation intensive so it is recommended to keep the pixel spacing high, for example 8.0 um. If the precision is not enough, for example when
        too many regions are reduced to single points, the pixel spacing can be lowered. To lower the number of generated control points in the polygon the epsilon parameter can
        be raised with lower pixel spacing. The epsilon is the maximal allowed distance of the contour if the region and the annotation lines. At a given pixel spacing a higher
        epsilon yields less precise annotation polygons with less control points. For details see: https://en.wikipedia.org/wiki/Ramer–Douglas–Peucker_algorithm

        Args:
            image (dptimagereader.ImageReader, str): Source mask image object or file path.
            spacing (float, None): Processing pixel spacing (micrometer). The lowest level is used if None.
            spacing_tolerance (float, None): Pixel spacing tolerance (percentage).
            label_map (dict): Mapping of the target annotation group names to label value lists from the mask. E.g. {'tumor': [1, 2], 'normal': [5, 6]}
            single_points (bool): Include annotations that are reduced to single points.
            rdp_epsilon (float): Maximum allowed distance (in pixels) of the contour of the outlined region from the annotation lines.

        Raises:
            NonLabelImageTypeError: The image format is not suitable for region outlining.
            InvalidCoordinateListError: The format of the coordinate list is invalid.
            InvalidAnnotationCoordinateListError: The length of the coordinate list does not match the type of the annotation.
            UnknownAnnotationTypeError: The annotation type in unknown.
            InvalidAnnotationColorError: The annotation color format is invalid.
        """

        # Open the mask image and check format.
        #
        input_image = image if isinstance(image, dptimagereader.ImageReader) else dptimagereader.ImageReader(image_path=image,
                                                                                                             spacing_tolerance=spacing_tolerance,
                                                                                                             input_channels=None,
                                                                                                             cache_path=None)

        if input_image.dtype not in (np.uint8, np.uint16, np.uint32) or input_image.coding != 'monochrome':
            raise dptimageerrors.NonLabelImageTypeError(input_image.dtype, input_image.coding, input_image.path)

        # Load the content of the image.
        #
        pixel_spacing = input_image.refine(spacing=spacing) if spacing is not None else input_image.spacings[0]
        content = input_image.content(spacing=pixel_spacing)
        content = content.squeeze()

        # Create label map if empty.
        #
        if label_map:
            label_groups = label_map
        else:
            # Add all labels except 0 for outlining, without being in any group.
            #
            label_values = np.unique(content).tolist()

            if 0 in label_values:
                label_values.remove(0)

            label_groups = {None: label_values}

        # Convert the regions in this group to annotations.
        #
        for group_name in label_groups:
            # The current label set.
            #
            labels = label_groups[group_name]

            # Select the regions with the current labels.
            #
            binary_content = np.isin(content, labels)

            # Label the content and identify the objects.
            #
            content_labels, _ = scipy.ndimage.measurements.label(input=binary_content)
            content_objects = scipy.ndimage.measurements.find_objects(input=content_labels)

            for object_index, object_slice in enumerate(content_objects):
                # Get an object patch.
                #
                object_patch = content_labels[object_slice]
                object_patch = np.equal(object_patch, object_index + 1)

                # Calculate contour.
                #
                object_patch_padded = np.pad(array=object_patch, pad_width=1, mode='constant', constant_values=0)
                object_patch_filled = scipy.ndimage.binary_fill_holes(input=object_patch_padded)
                object_patch_contour = skimage.measure.find_contours(array=object_patch_filled, level=0.5, fully_connected='high', positive_orientation='high')[0]

                # Simplify the list of points with Ramer-Douglas-Peucker algorithm.
                #
                points = rdp.rdp(M=object_patch_contour, epsilon=rdp_epsilon, dist=rdp.pldist, algo='iter', return_mask=False)

                # Correct the contour coordinates with crop start.
                #
                points[:, 0] += object_slice[0].start - 1
                points[:, 1] += object_slice[1].start - 1

                # Correct the point set to a single point if it is just 2 points.
                #
                if points.shape[0] < 3:
                    target_type = 'dot'
                    points = np.average(points, axis=0)
                else:
                    target_type = 'polygon'

                # Scale the pixel coordinates to spatial coordinates.
                #
                points *= pixel_spacing

                # Save the points.
                #
                if 3 <= points.shape[0] or single_points:
                    self.add(annotation=target_type, coordinates=points, name=None, group=group_name, color=None)

        # Close the input image.
        #
        if not isinstance(image, dptimagereader.ImageReader):
            input_image.close()

    def shift(self, vertical, horizontal):
        """
        Shift the annotations with the given vertical and horizontal shifts.
        Args:
            vertical (float): Vertical shift.
            horizontal (float): Horizontal shift.
        """

        for annotation_item in self.__annotations:
            for index in range(len(annotation_item['coordinates'])):
                coordinate = annotation_item['coordinates'][index]
                annotation_item['coordinates'][index] = (coordinate[0] + vertical, coordinate[1] + horizontal)

    @property
    def path(self):
        """
        Get the path of the opened annotation.

        Returns:
            str: Path of the opened annotation.
        """

        return self.__path

    @property
    def groups(self):
        """
        Get the groups.

        Returns:
            dict: Dictionary of groups with name as key.
        """

        return self.__groups.copy()

    @property
    def annotations(self):
        """
        Get the list of annotations.

        Returns:
            list: List of annotations.
        """

        return self.__annotations.copy()

    def counts(self):
        """
        Get the annotation counts per group.

        Returns:
            dict: Annotation counts per group (not counting the sub-groups).
        """

        annotation_counts = dict()

        for annotation_item in self.__annotations:
            if annotation_item['group'] not in annotation_counts:
                annotation_counts[annotation_item['group']] = 1
            else:
                annotation_counts[annotation_item['group']] += 1

        return annotation_counts

    def add(self, annotation, coordinates, name=None, group=None, color=None):
        """
        Insert annotation. If the group is unknown, a new group is created for it. The number of coordinates must match the type of the annotation. The None group is the root.

        Args:
            annotation (str): Annotation type. Valid types are 'dot', 'polygon', 'spline', 'points', 'measurement', and 'rectangle'.
            coordinates (np.ndarray, list): Numpy ndarray of shape (coordinates, 2) or list of (row, col) coordinate pairs that are spatial coordinates on the corresponding image.
            name (str, None): Name of the annotation. If None, a name is automatically generated.
            group (str, None): Name of the group.
            color (tuple, None): A (red, green, blue) tuple of the color of the annotation. If None the default color will be used.

        Raises:
            InvalidCoordinateListError: The format of the coordinate list is invalid.
            InvalidAnnotationCoordinateListError: The length of the coordinate list does not match the type of the annotation.
            UnknownAnnotationTypeError: The annotation type in unknown.
            InvalidAnnotationColorError: The annotation color format is invalid.
        """

        # Check the coordinate list.
        #
        if isinstance(coordinates, np.ndarray):
            if coordinates.ndim != 2 or coordinates.shape[1] != 2:
                raise dptimageerrors.InvalidCoordinateListError(coordinates)
        elif type(coordinates) != list and type(coordinates) != tuple or any(type(coordinate_item) != tuple or len(coordinate_item) != 2 for coordinate_item in coordinates):
            raise dptimageerrors.InvalidCoordinateListError(coordinates)

        # Check the annotation type.
        #
        coordinate_count = coordinates.shape[0] if isinstance(coordinates, np.ndarray) else len(coordinates)

        if annotation == 'dot':
            if coordinate_count != 1:
                raise dptimageerrors.InvalidAnnotationCoordinateListError(coordinate_count, annotation)

        elif annotation == 'polygon':
            if coordinate_count < 3:
                raise dptimageerrors.InvalidAnnotationCoordinateListError(coordinate_count, annotation)

        elif annotation == 'spline':
            if coordinate_count < 3:
                raise dptimageerrors.InvalidAnnotationCoordinateListError(coordinate_count, annotation)

        elif annotation == 'points':
            if coordinate_count == 0:
                raise dptimageerrors.InvalidAnnotationCoordinateListError(coordinate_count, annotation)

        elif annotation == 'measurement':
            if coordinate_count != 2:
                raise dptimageerrors.InvalidAnnotationCoordinateListError(coordinate_count, annotation)

        elif annotation == 'rectangle':
            if coordinate_count != 4:
                raise dptimageerrors.InvalidAnnotationCoordinateListError(coordinate_count, annotation)

        else:
            raise dptimageerrors.UnknownAnnotationTypeError(annotation)

        # Check color.
        #
        if color is not None and (type(color) != list and type(color) != tuple or len(color) != 3):
            raise dptimageerrors.InvalidAnnotationColorError(color)

        # Add annotation.
        #
        self.__annotations.append({'name': name if name is not None else 'Annotation {index}'.format(index=len(self.__annotations)),
                                   'type': annotation,
                                   'group': group,
                                   'color': tuple(color) if color else self.__default_annotation_color,
                                   'coordinates': [(coordinate_item[0], coordinate_item[1]) for coordinate_item in coordinates]})

        # Add to the group.
        #
        if group is not None and group not in self.__groups:
            self.__groups[group] = {'group': None, 'color': self.__default_group_color}

    def join(self, item, group):
        """
        Join an annotation or a group to a group. It is removed from its old group.

        Args:
            item (int, str): Index of the annotation or name of the group.
            group (str, None): Name of the group.

        Raises:
            InvalidAnnotationIndexError: The annotation index is invalid.
            UnknownAnnotationGroupError: The group name is invalid.
        """

        # Check if it is an annotation or group.
        #
        if type(item) == int:
            # Check if the index of the annotation is valid.
            #
            if item < 0 or len(self.__annotations) <= item:
                raise dptimageerrors.InvalidAnnotationIndexError(len(self.__annotations), item)

            # Add the annotation to the new group.
            #
            old_group = self.__annotations[item]['group']
            self.__annotations[item]['group'] = group

            # Create new group if necessary and remove the old one.
            #
            if group is not None and group not in self.__groups:
                self.__groups[group] = {'group': None, 'color': self.__default_group_color}

            self.__testgroup(group_name=old_group)

        else:
            # Check if the group exists.
            #
            if item not in self.__groups:
                raise dptimageerrors.UnknownAnnotationGroupError(list(self.__groups.keys()), item)

            # Set the new group.
            #
            old_group = self.__groups[item]['group']
            self.__groups[item]['group'] = group

            # Create new group if necessary and remove the old one.
            #
            if group is not None and group not in self.__groups:
                self.__groups[group] = {'group': None, 'color': self.__default_group_color}

            self.__testgroup(group_name=old_group)

    def remove(self, item):
        """
        Remove an annotation or group and delete its group if it is empty.

        Args:
            item (int, str): Index of the annotation or name of the group.

        Raises:
            InvalidAnnotationIndexError: The annotation index is invalid.
            UnknownAnnotationGroupError: The group name is invalid.
        """

        if type(item) == int:
            # Check if the index of the annotation is valid.
            #
            if item < 0 or len(self.__annotations) <= item:
                raise dptimageerrors.InvalidAnnotationIndexError(len(self.__annotations), item)

            # Remove the annotation.
            #
            removed_annotation = self.__annotations.pop(item)

            # Delete the old group if it is empty.
            #
            self.__testgroup(group_name=removed_annotation['group'])

        else:
            # Check if the group exists.
            #
            if item is not None and item not in self.__groups:
                raise dptimageerrors.UnknownAnnotationGroupError(list(self.__groups.keys()), item)

            groups_to_remove = {item}
            while groups_to_remove:
                group_name = groups_to_remove.pop()

                # Remove the group and check the parent group.
                #
                if group_name in self.__groups:
                    removed_group = self.__groups.pop(group_name)

                    self.__testgroup(group_name=removed_group['group'])

                # Remove all annotations in this group.
                #
                for annotation_index in reversed(range(len(self.__annotations))):
                    if self.__annotations[annotation_index]['group'] == group_name:
                        del self.__annotations[annotation_index]

                # Remove all groups in this group.
                #
                for group_item_name in self.__groups:
                    if self.__groups[group_item_name]['group'] == group_name:
                        groups_to_remove.add(group_item_name)

    def clear(self):
        """Clear all annotations and groups."""

        # Clear the object by removing the root group.
        #
        self.remove(item=None)
