"""
This module can compare to mask label images.
"""

from ..errors import processingerrors as dptprocessingerrors
from ..image import imagereader as dptimagereader
from ..image import imagewriter as dptimagewriter

import numpy as np
import scipy.ndimage
import logging
import datetime
import time
import os

#----------------------------------------------------------------------------------------------------

def _array_measures(array_0, label_0, array_1, label_1):
    """
    Calculate values for array similarity measurement: size of the region with the given labels in both arrays, union and intersection sizes.

    Args:
        array_0 (np.ndarray): First array.
        label_0 (int,): Label in the first array.
        array_1 (np.ndarray): Second array.
        label_1 (int): Label in the second array.

    Returns:
        (int, int, int, int): Region size in first array, region size in second array, union size, intersection size.

    Raises:
        ArrayShapeMismatch: The array shapes do not match.
    """

    # Compare the arrays.
    #
    if array_0.shape != array_1.shape:
        raise dptprocessingerrors.ArrayShapeMismatch(array_0.shape, array_1.shape)

    # Calculate the values.
    #
    bool_array_0 = np.equal(array_0, label_0)
    bool_array_1 = np.equal(array_1, label_1)
    bool_array_intersection = np.logical_and(bool_array_0, bool_array_1)
    bool_array_union = np.logical_or(bool_array_0, bool_array_1)

    # Return metrics.
    #
    return bool_array_0.astype(dtype=np.uint8).sum(), bool_array_1.astype(dtype=np.uint8).sum(), bool_array_union.astype(dtype=np.uint8).sum(), bool_array_intersection.astype(dtype=np.uint8).sum()

#----------------------------------------------------------------------------------------------------

def _array_classification(reference_array_binary, template_array_binary, detection_threshold, full_connectivity):
    """
    Classify the reference binary array into true positive and false negative detections.

    Args:
        reference_array_binary (np.ndarray): Reference binary array.
        template_array_binary (np.ndarray): Template binary array.
        detection_threshold (float): Detection ratio threshold. A region is considered as detected if larger than threshold area ratio of it is detected.
        full_connectivity (bool): Connectivity matrix. If true edge and point neighbors will be used otherwise edge neighbors only.

    Returns:
        np.ndarray, list, list: Labeled reference image, region bounding boxes, true positive region flags for each region.

    Raises:
        ArrayShapeMismatch: The array shapes do not match.
    """

    # Compare the array shapes.
    #
    if reference_array_binary.shape != template_array_binary.shape:
        raise dptprocessingerrors.ArrayShapeMismatch(reference_array_binary.shape, template_array_binary.shape)

    # Identify the reference objects.
    #
    connectivity_structure = np.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]] if full_connectivity else [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    reference_array_labels, _ = scipy.ndimage.measurements.label(input=reference_array_binary, structure=connectivity_structure)
    reference_array_objects = scipy.ndimage.measurements.find_objects(input=reference_array_labels)

    # Collect true positive and false negative regions.
    #
    true_positive_flags = []

    for reference_object_index in range(len(reference_array_objects)):
        reference_object_patch = reference_array_labels[reference_array_objects[reference_object_index]]
        reference_object_patch = np.equal(reference_object_patch, reference_object_index + 1)

        template_object_patch = template_array_binary[reference_array_objects[reference_object_index]]
        reference_object_detected = np.logical_and(reference_object_patch, template_object_patch)

        reference_object_size = np.sum(a=reference_object_patch, axis=None)
        reference_object_detected_size = np.sum(a=reference_object_detected, axis=None)

        is_true_positive_object = detection_threshold < reference_object_detected_size / reference_object_size
        true_positive_flags.append(is_true_positive_object)

    # Return the collected statistics.
    #
    return reference_array_labels, reference_array_objects, true_positive_flags

#----------------------------------------------------------------------------------------------------

def _array_comparision(reference_array_binary,
                       template_array_binary,
                       detection_threshold,
                       full_connectivity,
                       detected_positive_label,
                       detected_negative_label,
                       missed_positive_label,
                       missed_negative_label,
                       output_array=None):
    """
    Label the detected, missed and false positive regions by comparing the reference (mask) and the template (inference) arrays.

    Args:
        reference_array_binary (np.ndarray): Reference binary array.
        template_array_binary (np.ndarray): Template binary array.
        detection_threshold (float): Detection ratio threshold. A region is considered as detected if larger than threshold area ratio of it is detected.
        full_connectivity (bool): Connectivity matrix. If true edge and point neighbors will be used otherwise edge neighbors only.
        detected_positive_label (int): Label for parts of the detected reference regions that are overlapping with template regions.
        detected_negative_label (int): Label for parts of the detected reference regions that are not overlapping with template regions.
        missed_positive_label (int): Label for parts of the missed reference regions that are overlapping with template regions.
        missed_negative_label (int): Label for part of the missed reference regions that are not overlapping with template regions.
        output_array (np.ndarray, None): Output array. Allocated automatically if None. Only the non-zero parts are overwritten.

    Returns:
        np.ndarray: Result label array.

    Raises:
        ArrayShapeMismatch: The array shapes do not match.
    """

    # Calculate classification
    reference_array_labels, reference_array_objects, reference_true_positive_flags = _array_classification(reference_array_binary=reference_array_binary,
                                                                                                           template_array_binary=template_array_binary,
                                                                                                           detection_threshold=detection_threshold,
                                                                                                           full_connectivity=full_connectivity)

    # Allocate the result array.
    #
    result_array = np.zeros(shape=reference_array_binary.shape, dtype=np.uint8) if output_array is None else output_array

    # Go through the reference objects.
    #
    for reference_object_index in range(len(reference_array_objects)):
        # Get he reference object as binary array.
        #
        reference_object_bounding_box = reference_array_objects[reference_object_index]
        reference_object_patch = reference_array_labels[reference_object_bounding_box]
        reference_object_patch = np.equal(reference_object_patch, reference_object_index + 1)

        # Get the template object as binary array and identify the detected and missed parts of the reference object.
        #
        template_object_patch = template_array_binary[reference_object_bounding_box]
        reference_object_patch_detected = np.logical_and(reference_object_patch, template_object_patch)
        reference_object_patch_missed = np.logical_and(reference_object_patch, np.logical_not(template_object_patch))

        # Prepare the result part and identify the empty parts to prevent overwriting.
        #
        result_object_patch = result_array[reference_object_bounding_box]
        result_object_patch_empty = np.equal(result_object_patch, 0)

        # Color the reference object.
        #
        if reference_true_positive_flags[reference_object_index]:
            result_object_patch[np.logical_and(reference_object_patch_detected, result_object_patch_empty)] = detected_positive_label
            result_object_patch[np.logical_and(reference_object_patch_missed, result_object_patch_empty)] = detected_negative_label
        else:
            result_object_patch[np.logical_and(reference_object_patch_detected, result_object_patch_empty)] = missed_positive_label
            result_object_patch[np.logical_and(reference_object_patch_missed, result_object_patch_empty)] = missed_negative_label

    # Return the result array.
    #
    return result_array

#----------------------------------------------------------------------------------------------------

def array_similarity(reference_array, reference_label, template_array, template_label):
    """
    Compare two arrays and calculate either the Dice score and the Jaccard index.

    Args:
        reference_array (np.ndarray): Reference array.
        reference_label (int): Reference label.
        template_array (np.ndarray): Template array.
        template_label (int): Template label.

    Returns:
        (float, float): Dice score, Jaccard index.

    Raises:
        ArrayShapeMismatch: The array shapes do not match.
    """

    # Calculate values.
    #
    reference_size, template_size, union_size, intersection_size = _array_measures(array_0=reference_array, label_0=reference_label, array_1=template_array, label_1=template_label)

    # Calculate scores.
    #
    dice_score = 2.0 * intersection_size / (reference_size + template_size)
    jaccard_index = intersection_size / union_size

    # Return the calculated scores.
    #
    return dice_score, jaccard_index

#----------------------------------------------------------------------------------------------------

def image_similarity(reference, reference_label, template, template_label, pixel_spacing, spacing_tolerance):
    """
    Compare two images and calculate either the Dice score and the Jaccard index.

    Args:
        reference (dptimagereader.ImageReader, str): Reference image object or path.
        reference_label (int): Label of the region to compare in the reference image.
        template (dptimagereader.ImageReader, str): Template image object path.
        template_label (int): Label of the region to compare in the template image.
        pixel_spacing (float): Pixel spacing (micrometer).
        spacing_tolerance (float, None): Pixel spacing tolerance (percentage).

    Returns:
        (float, float): Calculated Dice score and Jaccard index.

    Raises:
        ReferenceTemplateImageShapeMismatchError: The reference and template image shapes do not match.
        ArrayShapeMismatch: Image sizes do not match.

        DigitalPathologyConfigError: Configuration errors.
        DigitalPathologyImageError: Image errors.
    """

    # Start time measurement.
    #
    start_time = time.time()

    # Log parameters.
    #
    logger = logging.getLogger(name=__name__)

    logger.info('Calculating image similarity...')
    logger.info('Reference image: {path}'.format(path=reference.path if isinstance(reference, dptimagereader.ImageReader) else reference))
    logger.info('Reference label: {label}'.format(label=reference_label))
    logger.info('Template image: {path}'.format(path=template.path if isinstance(template, dptimagereader.ImageReader) else template))
    logger.info('Template label: {label}'.format(label=template_label))
    logger.info('Pixel spacing: {spacing}'.format(spacing=pixel_spacing))

    # Open images.
    #
    reference_image = reference if isinstance(reference, dptimagereader.ImageReader) else dptimagereader.ImageReader(image_path=reference,
                                                                                                                     spacing_tolerance=spacing_tolerance,
                                                                                                                     input_channels=None,
                                                                                                                     cache_path=None)

    template_image = template if isinstance(template, dptimagereader.ImageReader) else dptimagereader.ImageReader(image_path=template,
                                                                                                                  spacing_tolerance=spacing_tolerance,
                                                                                                                  input_channels=None,
                                                                                                                  cache_path=None)

    # Check if the reference and template shapes match.
    #
    reference_level = reference_image.level(spacing=pixel_spacing)
    template_level = template_image.level(spacing=pixel_spacing)

    if reference_image.shapes[reference_level] != template_image.shapes[template_level]:
        raise dptprocessingerrors.ReferenceTemplateImageShapeMismatchError(reference_image.path, reference_image.shapes[reference_level], template_image.path, template_image.shapes[template_level])

    # Extract complete image content.
    #
    reference_content = reference_image.content(spacing=reference_image.spacings[reference_level]).squeeze()
    template_content = template_image.content(spacing=template_image.spacings[template_level]).squeeze()

    # Calculate values.
    #
    dice_score, jaccard_index = array_similarity(reference_array=reference_content, reference_label=reference_label, template_array=template_content, template_label=template_label)

    # Log the result.
    #
    logger.info('Dice score: {score:01.9f}'.format(score=dice_score))
    logger.info('Jaccard index: {score:01.9f}'.format(score=jaccard_index))

    # Close the reference and template images.
    #
    if isinstance(template, dptimagereader.ImageReader):
        template_image.close()

    if not isinstance(reference, dptimagereader.ImageReader):
        reference_image.close()

    # Log execution time.
    #
    exec_time = time.time() - start_time
    logger.debug('Done in {delta}'.format(delta=datetime.timedelta(seconds=exec_time)))

    # Return the calculated scores.
    #
    return dice_score, jaccard_index

#----------------------------------------------------------------------------------------------------

def array_statistics(reference_array, reference_label, template_array, template_label, true_detection_threshold, false_detection_threshold, full_connectivity):
    """
    Calculate array statistics: true positive (detected), false negative (missed), false positive (falsely detected) region counts.

    Args:
        reference_array (np.ndarray): Reference array.
        reference_label (int): Reference label.
        template_array (np.ndarray): Template array.
        template_label (int): Template label.
        true_detection_threshold (float): A reference region is detected (TP) if larger than threshold ratio of it overlaps with the template, otherwise missed (FN).
        false_detection_threshold (float): A template region is a detection if larger than threshold area ratio of it overlaps with the reference, otherwise false detection (FP).
        full_connectivity (bool): Connectivity matrix. If true edge and point neighbors will be used otherwise edge neighbors only.

    Returns:
        (int, int, int): True positive, false negative, false positive region counts.

    Raises:
        ArrayShapeMismatch: The array shapes do not match.
    """

    # Binarize the arrays.
    #
    reference_array_binary = np.equal(reference_array, reference_label)
    template_array_binary = np.equal(template_array, template_label)

    # Calculate statistics.
    #
    _, _, true_positive_flags = _array_classification(reference_array_binary=reference_array_binary,
                                                      template_array_binary=template_array_binary,
                                                      detection_threshold=true_detection_threshold,
                                                      full_connectivity=full_connectivity)

    _, _, not_false_positive_flags = _array_classification(reference_array_binary=template_array_binary,
                                                           template_array_binary=reference_array_binary,
                                                           detection_threshold=false_detection_threshold,
                                                           full_connectivity=full_connectivity)

    # Summarize statistics.
    #
    true_positives = true_positive_flags.count(True)
    false_negatives = len(true_positive_flags) - true_positives
    false_positives = not_false_positive_flags.count(False)

    # Return the calculated statistics.
    #
    return true_positives, false_negatives, false_positives

#----------------------------------------------------------------------------------------------------

def image_statistics(reference, reference_label, template, template_label, pixel_spacing, spacing_tolerance, true_detection_threshold, false_detection_threshold, full_connectivity):
    """
    Calculate image statistics: true positive (detected), false negative (missed), false positive (falsely detected) region counts.

    Args:
        reference (dptimagereader.ImageReader, str): Reference image object or path.
        reference_label (int): Label of the region to compare in the reference image.
        template (dptimagereader.ImageReader, str): Template image object path.
        template_label (int): Label of the region to compare in the template image.
        pixel_spacing (float): Pixel spacing (micrometer).
        spacing_tolerance (float, None): Pixel spacing tolerance (percentage).
        true_detection_threshold (float): A reference region is detected (TP) if larger than threshold ratio of it overlaps with the template, otherwise missed (FN).
        false_detection_threshold (float): A template region is a detection if larger than threshold area ratio of it overlaps with the reference, otherwise false detection (FP).
        full_connectivity (bool): Connectivity matrix. If true edge and point neighbors will be used otherwise edge neighbors only.

    Returns:
        (int, int, int): True positive, false negative, false positive region counts.

    Raises:
        ReferenceTemplateImageShapeMismatchError: The reference and template image shapes do not match.
        ArrayShapeMismatch: The array shapes do not match.

        DigitalPathologyConfigError: Configuration errors.
        DigitalPathologyImageError: Image errors.
    """

    # Start time measurement.
    #
    start_time = time.time()

    # Log parameters.
    #
    logger = logging.getLogger(name=__name__)

    logger.info('Calculating image statistics...')
    logger.info('Reference image: {path}'.format(path=reference.path if isinstance(reference, dptimagereader.ImageReader) else reference))
    logger.info('Reference label: {label}'.format(label=reference_label))
    logger.info('Template image: {path}'.format(path=template.path if isinstance(template, dptimagereader.ImageReader) else template))
    logger.info('Template label: {label}'.format(label=template_label))
    logger.info('Pixel spacing: {spacing}'.format(spacing=pixel_spacing))
    logger.info('True detection threshold: {threshold}'.format(threshold=true_detection_threshold))
    logger.info('False Detection threshold: {threshold}'.format(threshold=false_detection_threshold))
    logger.info('Full connectivity: {flag}'.format(flag=full_connectivity))

    # Open images.
    #
    reference_image = reference if isinstance(reference, dptimagereader.ImageReader) else dptimagereader.ImageReader(image_path=reference,
                                                                                                                     spacing_tolerance=spacing_tolerance,
                                                                                                                     input_channels=None,
                                                                                                                     cache_path=None)

    template_image = template if isinstance(template, dptimagereader.ImageReader) else dptimagereader.ImageReader(image_path=template,
                                                                                                                  spacing_tolerance=spacing_tolerance,
                                                                                                                  input_channels=None,
                                                                                                                  cache_path=None)

    # Check if the reference and template shapes match.
    #
    reference_level = reference_image.level(spacing=pixel_spacing)
    template_level = template_image.level(spacing=pixel_spacing)

    if reference_image.shapes[reference_level] != template_image.shapes[template_level]:
        raise dptprocessingerrors.ReferenceTemplateImageShapeMismatchError(reference_image.path, reference_image.shapes[reference_level], template_image.path, template_image.shapes[template_level])

    # Load image contents.
    #
    reference_content = reference_image.content(spacing=reference_image.spacings[reference_level]).squeeze()
    template_content = template_image.content(spacing=template_image.spacings[template_level]).squeeze()

    # Calculate statistics.
    #
    true_positives, false_negatives, false_positives = array_statistics(reference_array=reference_content,
                                                                        reference_label=reference_label,
                                                                        template_array=template_content,
                                                                        template_label=template_label,
                                                                        true_detection_threshold=true_detection_threshold,
                                                                        false_detection_threshold=false_detection_threshold,
                                                                        full_connectivity=full_connectivity)

    # Log the result.
    #
    logger.info('True positives: {count}'.format(count=true_positives))
    logger.info('False negatives: {count}'.format(count=false_negatives))
    logger.info('False positives: {count}'.format(count=false_positives))

    # Close the reference and template images.
    #
    if isinstance(template, dptimagereader.ImageReader):
        template_image.close()

    if not isinstance(reference, dptimagereader.ImageReader):
        reference_image.close()

    # Log execution time.
    #
    exec_time = time.time() - start_time
    logger.debug('Done in {delta}'.format(delta=datetime.timedelta(seconds=exec_time)))

    # Return the calculated statistics.
    #
    return true_positives, false_negatives, false_positives

#----------------------------------------------------------------------------------------------------

def array_comparision(reference_array,
                      reference_label,
                      template_array,
                      template_label,
                      true_detection_threshold,
                      false_detection_threshold,
                      full_connectivity,
                      detected_positive_label=1,
                      detected_negative_label=2,
                      missed_positive_label=3,
                      missed_negative_label=4,
                      false_positive_label=5,
                      false_negative_label=6,
                      output_array=None):
    """
    Label the detected, missed and false positive regions by comparing the reference (mask) and the template (inference) arrays.

    Args:
        reference_array (np.ndarray): Reference array.
        reference_label (int): Reference label value. Only this label is used for comparison.
        template_array (np.ndarray): Template array.
        template_label (int): Template label value. Only this label is used for comparison.
        true_detection_threshold (float): A reference region is detected (TP) if larger than threshold ratio of it overlaps with the template, otherwise missed (FN).
        false_detection_threshold (float): A template region is a detection if larger than threshold area ratio of it overlaps with the reference, otherwise false detection (FP).
        full_connectivity (bool): Connectivity matrix. If true edge and point neighbors will be used otherwise edge neighbors only.
        detected_positive_label (int): Label for parts of the detected reference regions that are overlapping with template regions.
        detected_negative_label (int): Label for parts of the detected reference regions that are not overlapping with template regions.
        missed_positive_label (int): Label for parts of the missed reference regions that are overlapping with template regions.
        missed_negative_label (int): Label for part of the missed reference regions that are not overlapping with template regions.
        false_positive_label (int): Part of false positive template regions that are overlapping with reference regions.
        false_negative_label (int): Part of false positive template regions that are not overlapping with reference regions.
        output_array (np.ndarray, None): Output array. Allocated automatically if None. Only the non-zero parts are overwritten.

    Returns:
        np.ndarray: Result label array.

    Raises:
        ArrayShapeMismatch: The array shapes do not match.
    """

    # Select the reference and template labels.
    #
    reference_content_binary = np.equal(reference_array, reference_label)
    template_content_binary = np.equal(template_array, template_label)

    # Label the output array.
    #
    output_array = _array_comparision(reference_array_binary=reference_content_binary,
                                      template_array_binary=template_content_binary,
                                      detection_threshold=true_detection_threshold,
                                      full_connectivity=full_connectivity,
                                      detected_positive_label=detected_positive_label,
                                      detected_negative_label=detected_negative_label,
                                      missed_positive_label=missed_positive_label,
                                      missed_negative_label=missed_negative_label,
                                      output_array=output_array)

    output_array = _array_comparision(reference_array_binary=template_content_binary,
                                      template_array_binary=reference_content_binary,
                                      detection_threshold=false_detection_threshold,
                                      full_connectivity=full_connectivity,
                                      detected_positive_label=0,
                                      detected_negative_label=false_positive_label,
                                      missed_positive_label=0,
                                      missed_negative_label=false_negative_label,
                                      output_array=output_array)

    # Return the result array.
    #
    return output_array

#----------------------------------------------------------------------------------------------------

def image_comparison(reference,
                     reference_label,
                     template,
                     template_label,
                     pixel_spacing,
                     spacing_tolerance,
                     true_detection_threshold,
                     false_detection_threshold,
                     full_connectivity,
                     output_path,
                     overwrite,
                     detected_positive_label=1,
                     detected_negative_label=2,
                     missed_positive_label=3,
                     missed_negative_label=4,
                     false_positive_label=5,
                     false_negative_label=6):
    """
    Label the detected, missed and false positive regions by comparing the reference (mask) and the template (inference) images.

    Args:
        reference (dptimagereader.ImageReader, str): Reference image object or path.
        reference_label (int): Label of the region to compare in the reference image.
        template (dptimagereader.ImageReader, str): Template image object path.
        template_label (int): Label of the region to compare in the template image.
        pixel_spacing (float): Pixel spacing (micrometer).
        spacing_tolerance (float, None): Pixel spacing tolerance (percentage).
        output_path (str): Output image path.
        overwrite (bool): Overwrite flag.
        true_detection_threshold (float): A reference region is detected (TP) if larger than threshold ratio of it overlaps with the template, otherwise missed (FN).
        false_detection_threshold (float): A template region is a detection if larger than threshold area ratio of it overlaps with the reference, otherwise false detection (FP).
        full_connectivity (bool): Connectivity matrix. If true edge and point neighbors will be used otherwise edge neighbors only.
        detected_positive_label (int): Label for parts of the detected reference regions that are overlapping with template regions.
        detected_negative_label (int): Label for parts of the detected reference regions that are not overlapping with template regions.
        missed_positive_label (int): Label for parts of the missed reference regions that are overlapping with template regions.
        missed_negative_label (int): Label for part of the missed reference regions that are not overlapping with template regions.
        false_positive_label (int): Part of false positive template regions that are overlapping with reference regions.
        false_negative_label (int): Part of false positive template regions that are not overlapping with reference regions.

    Raises:
        ReferenceTemplateImageShapeMismatchError: The reference and template image shapes do not match.
        ArrayShapeMismatch: The array shapes do not match.

        DigitalPathologyConfigError: Configuration errors.
        DigitalPathologyImageError: Image errors.
    """

    # Start time measurement.
    #
    start_time = time.time()

    # Log parameters.
    #
    logger = logging.getLogger(name=__name__)

    logger.info('Comparing regions...')
    logger.info('Reference image: {path}'.format(path=reference.path if isinstance(reference, dptimagereader.ImageReader) else reference))
    logger.info('Reference label: {label}'.format(label=reference_label))
    logger.info('Template image: {path}'.format(path=template.path if isinstance(template, dptimagereader.ImageReader) else template))
    logger.info('Template label: {label}'.format(label=template_label))
    logger.info('Pixel spacing: {spacing}'.format(spacing=pixel_spacing))
    logger.info('Output image: {path}'.format(path=output_path))
    logger.info('True detection threshold: {threshold}'.format(threshold=true_detection_threshold))
    logger.info('False Detection threshold: {threshold}'.format(threshold=false_detection_threshold))
    logger.info('Full connectivity: {flag}'.format(flag=full_connectivity))

    # Check if the target image already exits.
    #
    if not os.path.isfile(output_path) or overwrite:
        # Open images.
        #
        reference_image = reference if isinstance(reference, dptimagereader.ImageReader) else dptimagereader.ImageReader(image_path=reference,
                                                                                                                         spacing_tolerance=spacing_tolerance,
                                                                                                                         input_channels=None,
                                                                                                                         cache_path=None)

        template_image = template if isinstance(template, dptimagereader.ImageReader) else dptimagereader.ImageReader(image_path=template,
                                                                                                                      spacing_tolerance=spacing_tolerance,
                                                                                                                      input_channels=None,
                                                                                                                      cache_path=None)

        # Check if the reference and template shapes match.
        #
        reference_level = reference_image.level(spacing=pixel_spacing)
        template_level = template_image.level(spacing=pixel_spacing)

        if reference_image.shapes[reference_level] != template_image.shapes[template_level]:
            raise dptprocessingerrors.ReferenceTemplateImageShapeMismatchError(reference_image.path,
                                                                               reference_image.shapes[reference_level],
                                                                               template_image.path,
                                                                               template_image.shapes[template_level])

        # Extract complete image contents.
        #
        reference_content = reference_image.content(spacing=reference_image.spacings[reference_level]).squeeze()
        template_content = template_image.content(spacing=template_image.spacings[template_level]).squeeze()

        # Label the output array.
        #
        output_array = array_comparision(reference_array=reference_content,
                                         reference_label=reference_label,
                                         template_array=template_content,
                                         template_label=template_label,
                                         true_detection_threshold=true_detection_threshold,
                                         false_detection_threshold=false_detection_threshold,
                                         full_connectivity=full_connectivity,
                                         detected_positive_label=detected_positive_label,
                                         detected_negative_label=detected_negative_label,
                                         missed_positive_label=missed_positive_label,
                                         missed_negative_label=missed_negative_label,
                                         false_positive_label=false_positive_label,
                                         false_negative_label=false_negative_label,
                                         output_array=None)

        # Configure the image writer.
        #
        tile_size = 512
        image_writer = dptimagewriter.ImageWriter(image_path=output_path,
                                                  shape=reference_image.shapes[reference_level],
                                                  spacing=reference_image.spacings[reference_level],
                                                  dtype=np.uint8,
                                                  coding='monochrome',
                                                  compression=None,
                                                  interpolation=None,
                                                  tile_size=tile_size,
                                                  jpeg_quality=None,
                                                  empty_value=0,
                                                  skip_empty=None,
                                                  cache_path=None)

        # Save result.
        #
        output_array = output_array[None, ...]
        image_writer.fill(content=output_array)
        image_writer.close()

        # Close the reference and template images.
        #
        if isinstance(template, dptimagereader.ImageReader):
            template_image.close()

        if not isinstance(reference, dptimagereader.ImageReader):
            reference_image.close()

        # Log execution time.
        #
        exec_time = time.time() - start_time
        logger.debug('Done in {delta}'.format(delta=datetime.timedelta(seconds=exec_time)))

    else:
        logger.info('Skipping, target file already exits: {path}'.format(path=output_path))
