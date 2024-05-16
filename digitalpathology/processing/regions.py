"""
This module can filter regions or holes in regions in a given image.
"""

from ..errors import processingerrors as dptprocessingerrors
from ..image import imagereader as dptimagereader
from ..image import imagewriter as dptimagewriter

import numpy as np
import scipy.ndimage
import logging
import os
import datetime
import time
import math

#----------------------------------------------------------------------------------------------------

def _array_filtering(binary_array, diagonal_threshold, full_connectivity):
    """
    Classify the regions in the reference binary array into smaller and larger than threshold size diameter classes.

    Args:
        binary_array (np.ndarray): Input binary array.
        diagonal_threshold (float): Region diagonal low threshold (pixels).
        full_connectivity (bool): Connectivity matrix. If true edge and point neighbors will be used otherwise edge neighbors only.

    Returns:
        np.ndarray, list, list: Labeled array, region bounding boxes, large enough region flags for each region.
    """

    # Identify the reference objects.
    #
    connectivity_structure = np.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]] if full_connectivity else [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    array_labels, _ = scipy.ndimage.measurements.label(input=binary_array, structure=connectivity_structure)
    array_objects = scipy.ndimage.measurements.find_objects(input=array_labels)

    # Collect true positive and false negative regions.
    #
    large_enough_flags = []

    for object_index in range(len(array_objects)):
        object_bounding_box = array_objects[object_index]

        object_height = object_bounding_box[0].stop - object_bounding_box[0].start
        object_width = object_bounding_box[1].stop - object_bounding_box[1].start
        object_diagonal = math.sqrt(object_height * object_height + object_width * object_width)

        large_enough_region = diagonal_threshold < object_diagonal
        large_enough_flags.append(large_enough_region)

    # Return the collected regions and flags.
    #
    return array_labels, array_objects, large_enough_flags

#----------------------------------------------------------------------------------------------------

def filter_regions_array(input_array, diagonal_threshold, full_connectivity, foreground_labels=None, background_label=0):
    """
    Filter out regions that have smaller than the configured diagonal. The function modifies the input array.

    Args:
        input_array (np.ndarray): Input array. Modified.
        diagonal_threshold (float): Region diagonal low threshold (pixels).
        full_connectivity (bool): Connectivity matrix. If true edge and point neighbors will be used otherwise edge neighbors only.
        foreground_labels (list, None): List of labels to consider as foreground, everything else is background. If empty every nonzero value is foreground.
        background_label (int): Label value for overwriting the identified small regions.

    Returns:
        np.ndarray, int, int: Result array, identified region count, filtered region count.
    """

    # Select the foreground labels from the array.
    #
    input_array_binary = np.isin(element=input_array, test_elements=foreground_labels) if foreground_labels is not None else input_array

    # Select the foreground labels from the array.
    #
    # Identify the objects.
    #
    array_labels, array_objects, large_enough_flags = _array_filtering(binary_array=input_array_binary, diagonal_threshold=diagonal_threshold, full_connectivity=full_connectivity)

    # Go through the objects and remove smaller than threshold regions.
    #
    removed_regions = 0
    for object_index in range(len(array_objects)):
        if not large_enough_flags[object_index]:
            object_bounding_box = array_objects[object_index]

            object_patch = array_labels[object_bounding_box]
            object_mask = np.equal(object_patch, object_index + 1)

            content_patch = input_array[object_bounding_box]
            content_patch[object_mask] = background_label

            removed_regions += 1

    # Return the result image, the total number of identified regions and the number of removed regions.
    #
    return input_array, len(array_objects), removed_regions

#----------------------------------------------------------------------------------------------------

def filter_regions_image(image,
                         output_path,
                         diagonal_threshold,
                         full_connectivity,
                         pixel_spacing,
                         spacing_tolerance,
                         foreground_labels=None,
                         background_label=0,
                         copy_path=None,
                         work_path=None,
                         clear_cache=False,
                         overwrite=True):
    """
    Filter out regions that have smaller than the configured diagonal. The image is loaded to memory as a whole. Large images may not fit into memory.

    Args:
        image (dptimagereader.ImageReader, str): Input image object or path.
        output_path (str): Output image path.
        diagonal_threshold (float): Region diagonal low threshold (micrometer).
        full_connectivity (bool): Connectivity matrix. If true edge and point neighbors will be used otherwise edge neighbors only.
        pixel_spacing (float, None): Processing pixel spacing (micrometer). If None, the lowest level is used.
        spacing_tolerance (float): Pixel spacing tolerance (percentage).
        foreground_labels (list, None): List of labels to consider as foreground, everything else is background. If empty every nonzero value is foreground.
        background_label (int): Label value for overwriting the identified small regions.
        copy_path (str, None): Copy directory path where the images and masks are cached before processing.
        work_path (str, None): Work directory path where the output is cached first before writing it to the target path.
        clear_cache (bool): Remove the cached images.
        overwrite (bool): If true existing targets will be overwritten.

    Raises:
        NonMonochromeInputImageError: The image is not monochrome.
        NonIntegralImageDataTypeError: The data type of the image is not integral.

        DigitalPathologyImageError: Image errors.
    """

    # Start time measurement.
    #
    start_time = time.time()

    # Log parameters.
    #
    logger = logging.getLogger(name=__name__)

    logger.info('Filtering small regions...')
    logger.info('Input image: {path}'.format(path=image.path if isinstance(image, dptimagereader.ImageReader) else image))
    logger.info('Result image: {path}'.format(path=output_path))
    logger.info('Diagonal threshold: {threshold} um'.format(threshold=diagonal_threshold))
    if pixel_spacing is not None:
        logger.info('Pixel spacing: {spacing} um'.format(spacing=pixel_spacing))
    else:
        logger.info('Pixel spacing: at lowest level')

    # Check if the target image already exits.
    #
    if not os.path.isfile(output_path) or overwrite:
        # Open image.
        #
        input_image = image if isinstance(image, dptimagereader.ImageReader) else dptimagereader.ImageReader(image_path=image,
                                                                                                             spacing_tolerance=spacing_tolerance,
                                                                                                             input_channels=None,
                                                                                                             cache_path=copy_path)

        # Check if the image is monochrome. Cannot filter color images.
        #
        if input_image.coding != 'monochrome':
            raise dptprocessingerrors.NonMonochromeInputImageError(input_image.path, input_image.coding)

        # Check if the image has integral data type. Hole filling works on binary images.
        #
        if input_image.dtype not in [np.uint8, np.uint16, np.uint32]:
            raise dptprocessingerrors.NonIntegralImageDataTypeError(input_image.path, input_image.dtype)

        # Identify processing level.
        #
        processing_level = input_image.level(spacing=pixel_spacing) if pixel_spacing is not None else 0
        logger.debug('Identified processing level: {level}'.format(level=processing_level))

        # Calculate size threshold.
        #
        pixel_threshold = diagonal_threshold / input_image.spacings[0]
        logger.debug('Diagonal low threshold: {threshold} pixels'.format(threshold=pixel_threshold))

        # Read image content into memory.
        #
        image_content = input_image.content(spacing=input_image.spacings[processing_level]).squeeze()
        logger.debug('Processing: {path}'.format(path=input_image.path))

        # Filter regions.
        #
        image_content, region_count, removed_regions = filter_regions_array(input_array=image_content,
                                                                            diagonal_threshold=pixel_threshold,
                                                                            full_connectivity=full_connectivity,
                                                                            foreground_labels=foreground_labels,
                                                                            background_label=background_label)

        # Log number of removed regions.
        #
        logger.debug('Removed: {removed} of {total} regions'.format(removed=removed_regions, total=region_count))

        # Configure the image writer.
        #
        tile_size = 512
        image_writer = dptimagewriter.ImageWriter(image_path=output_path,
                                                  shape=input_image.shapes[processing_level],
                                                  spacing=input_image.spacings[processing_level],
                                                  dtype=input_image.dtype,
                                                  coding=input_image.coding,
                                                  compression=None,
                                                  interpolation=None,
                                                  tile_size=tile_size,
                                                  jpeg_quality=None,
                                                  empty_value=0,
                                                  skip_empty=None,
                                                  cache_path=work_path)

        # Save result.
        #
        image_content = image_content[None, ...]
        image_writer.fill(content=image_content)

        # Finalize the output image and close the input image.
        #
        image_writer.close(clear=clear_cache)

        if not isinstance(image, dptimagereader.ImageReader):
            input_image.close(clear=clear_cache)

        # Log execution time.
        #
        exec_time = time.time() - start_time
        logger.debug('Done in {delta}'.format(delta=datetime.timedelta(seconds=exec_time)))

    else:
        logger.info('Skipping, target file already exits: {path}'.format(path=output_path))

#----------------------------------------------------------------------------------------------------

def filter_regions_batch(job_list,
                         diagonal_threshold,
                         full_connectivity,
                         pixel_spacing,
                         spacing_tolerance,
                         foreground_labels=None,
                         background_label=0,
                         copy_path=None,
                         work_path=None,
                         clear_cache=True,
                         overwrite=True):
    """
    Filter out regions that have smaller than the configured diagonal in bach. The images are loaded to memory as a whole. Large images may not fit into memory.

    Args:
        job_list (list): List of job (input path, output path) tuples.
        diagonal_threshold (float): Region diagonal low threshold (micrometer).
        full_connectivity (bool): Connectivity matrix. If true edge and point neighbors will be used otherwise edge neighbors only.
        pixel_spacing (float, None): Processing pixel spacing (micrometer). If None, the lowest level is used.
        spacing_tolerance (float): Pixel spacing tolerance (percentage).
        foreground_labels (list, None): List of labels to consider as foreground, everything else is background. If empty every nonzero value is foreground.
        background_label (int): Label value for overwriting the identified small regions.
        copy_path (str, None): Copy directory path where the images and masks are cached before processing.
        work_path (str, None): Work directory path where the output is cached first before writing it to the target path.
        clear_cache (bool): Remove the cached images.
        overwrite (bool): If true existing targets will be overwritten.

    Returns:
        (list, list): List of successfully processed items and list of failed items.
    """

    # Start time measurement.
    #
    start_time = time.time()

    # Log batch processing.
    #
    logger = logging.getLogger(name=__name__)

    logger.info('Filtering regions in batch mode...')
    logger.info('Job count: {count}'.format(count=len(job_list)))

    # Go through the job list.
    #
    failed_collection = []
    successful_collection = []
    for job_index in range(len(job_list)):
        # Get one job item.
        #
        input_path, output_path = job_list[job_index]

        try:
            # Print data paths if the mode is not a single file mode.
            #
            logger.info('Processing [{index}/{count}]: {path}'.format(index=job_index+1, count=len(job_list), path=input_path))

            # Filter the regions.
            #
            filter_regions_image(image=input_path,
                                 output_path=output_path,
                                 diagonal_threshold=diagonal_threshold,
                                 pixel_spacing=pixel_spacing,
                                 spacing_tolerance=spacing_tolerance,
                                 full_connectivity=full_connectivity,
                                 foreground_labels=foreground_labels,
                                 background_label=background_label,
                                 copy_path=copy_path,
                                 work_path=work_path,
                                 clear_cache=clear_cache,
                                 overwrite=overwrite)

        except Exception as exception:
            # Add case to the error collection.
            #
            failed_collection.append(output_path)
            logger.error('Error: {exception}'.format(exception=exception))

        else:
            # Save the result to the list of successful thresholds.
            #
            successful_collection.append(output_path)

    # Log execution time.
    #
    execution_time = time.time() - start_time
    logger.debug('Region filtering batch done in {delta}'.format(delta=datetime.timedelta(seconds=execution_time)))

    # Return a list of successful filterings.
    #
    return successful_collection, failed_collection

#----------------------------------------------------------------------------------------------------

def fill_holes_array(input_array, diagonal_threshold, full_connectivity, foreground_labels=None, fill_value=1):
    """
    Fill holes that have smaller than the configured diagonal. The function modifies the input array.

    Args:
        input_array (np.ndarray): Input array. Modified.
        diagonal_threshold (float): Region diagonal low threshold (pixels).
        full_connectivity (bool): Connectivity matrix. If true edge and point neighbors will be used otherwise edge neighbors only.
        foreground_labels (list, None): List of labels to consider as foreground, everything else is background. If empty every nonzero value is foreground.
        fill_value (int): Label value for overwriting the identified region holes.

    Returns:
        np.ndarray, int, int: Result array, identified region count, filtered region count.
    """

    # Select the background labels from the array.
    #
    input_array_binary = np.logical_not(np.isin(element=input_array, test_elements=foreground_labels)) if foreground_labels is not None else np.logical_not(input_array)

    # Identify the objects.
    #
    array_labels, array_objects, large_enough_flags = _array_filtering(binary_array=input_array_binary, diagonal_threshold=diagonal_threshold, full_connectivity=full_connectivity)

    # Go through the objects and fill smaller than threshold holes.
    #
    filled_holes = 0
    for object_index in range(len(array_objects)):
        if not large_enough_flags[object_index]:
            object_bounding_box = array_objects[object_index]

            object_patch = array_labels[object_bounding_box]
            object_mask = np.equal(object_patch, object_index + 1)

            content_patch = input_array[object_bounding_box]
            content_patch[object_mask] = fill_value

            filled_holes += 1

    # Return the result image, the total number of identified holes and the number of filled holes.
    #
    return input_array, len(array_objects), filled_holes

#----------------------------------------------------------------------------------------------------

def fill_holes_image(image,
                     output_path,
                     diagonal_threshold,
                     full_connectivity,
                     pixel_spacing,
                     spacing_tolerance,
                     foreground_labels=None,
                     fill_value=1,
                     copy_path=None,
                     work_path=None,
                     clear_cache=True,
                     overwrite=True):
    """
    Fill holes that have smaller than the configured diagonal. The image is loaded to memory as a whole. Large images may not fit into memory.

    Args:
        image (dptimagereader.ImageReader, str): Input image object or path.
        output_path (str): Output image path. The zoomed image.
        diagonal_threshold (float): Region diagonal low threshold (micrometer).
        full_connectivity (bool): Connectivity matrix. If true edge and point neighbors will be used otherwise edge neighbors only.
        pixel_spacing (float, None): Processing pixel spacing (micrometer). If None, the lowest level is used.
        spacing_tolerance (float): Pixel spacing tolerance (percentage).
        foreground_labels (list, None): List of labels to consider as foreground, everything else is background. If empty every nonzero value is foreground.
        fill_value (int): Label value for overwriting the identified region holes.
        copy_path (str, None): Copy directory path where the images and masks are cached before processing.
        work_path (str, None): Work directory path where the output is cached first before writing it to the target path.
        clear_cache (bool): Remove the cached images.
        overwrite (bool): If true existing targets will be overwritten.

    Raises:
        NonMonochromeInputImageError: The image is not monochrome.
        NonIntegralImageDataTypeError: The data type of the image is not integral.

        DigitalPathologyImageError: Image errors.
    """

    # Start time measurement.
    #
    start_time = time.time()

    # Log parameters.
    #
    logger = logging.getLogger(name=__name__)

    logger.info('Filling holes...')
    logger.info('Input image: {path}'.format(path=image.path if isinstance(image, dptimagereader.ImageReader) else image))
    logger.info('Result image: {path}'.format(path=output_path))
    logger.info('Diagonal threshold: {threshold} um'.format(threshold=diagonal_threshold))
    if pixel_spacing is not None:
        logger.info('Pixel spacing: {spacing} um'.format(spacing=pixel_spacing))
    else:
        logger.info('Pixel spacing: at lowest level')

    # Check if the target image already exits.
    #
    if not os.path.isfile(output_path) or overwrite:
        # Open image.
        #
        input_image = image if isinstance(image, dptimagereader.ImageReader) else dptimagereader.ImageReader(image_path=image,
                                                                                                             spacing_tolerance=spacing_tolerance,
                                                                                                             input_channels=None,
                                                                                                             cache_path=copy_path)

        # Check if the image is monochrome. Cannot filter color images.
        #
        if input_image.coding != 'monochrome':
            raise dptprocessingerrors.NonMonochromeInputImageError(input_image.path, input_image.coding)

        # Check if the image has integral data type. Hole filling works on binary images.
        #
        if input_image.dtype not in [np.uint8, np.uint16, np.uint32]:
            raise dptprocessingerrors.NonIntegralImageDataTypeError(input_image.path, input_image.dtype)

        if any(level_spacing is None for level_spacing in input_image.spacings):
            input_image.correct(spacing=1.0, level=0)

        # Calculate size threshold.
        #
        pixel_threshold = diagonal_threshold / input_image.spacings[0]
        logger.debug('Diagonal low threshold: {threshold} pixels'.format(threshold=pixel_threshold))

        # Identify processing level.
        #
        processing_level = input_image.level(spacing=pixel_spacing) if pixel_spacing is not None else 0
        logger.debug('Identified processing level: {level}'.format(level=processing_level))

        # Read image content into memory.
        #
        image_content = input_image.content(spacing=input_image.spacings[processing_level]).squeeze()
        logger.debug('Processing: {path}'.format(path=input_image.path))

        # Fill the holes.
        #
        image_content, hole_count, filled_holes = fill_holes_array(input_array=image_content,
                                                                   diagonal_threshold=pixel_threshold,
                                                                   full_connectivity=full_connectivity,
                                                                   foreground_labels=foreground_labels,
                                                                   fill_value=fill_value)

        # Log number of filled holes.
        #
        logger.debug('Filled: {filled} of {total} holes'.format(filled=filled_holes, total=hole_count))

        # Configure the image writer.
        #
        tile_size = 512
        image_writer = dptimagewriter.ImageWriter(image_path=output_path,
                                                  shape=input_image.shapes[processing_level],
                                                  spacing=input_image.spacings[processing_level],
                                                  dtype=input_image.dtype,
                                                  coding=input_image.coding,
                                                  compression=None,
                                                  interpolation=None,
                                                  tile_size=tile_size,
                                                  jpeg_quality=None,
                                                  empty_value=0,
                                                  skip_empty=None,
                                                  cache_path=work_path)

        # Save result.
        #
        image_content = image_content[None, ...]
        image_writer.fill(content=image_content)

        # Finalize the output image and close the input image.
        #
        image_writer.close(clear=clear_cache)

        if not isinstance(image, dptimagereader.ImageReader):
            input_image.close(clear=clear_cache)

        # Log execution time.
        #
        exec_time = time.time() - start_time
        logger.debug('Done in {delta}'.format(delta=datetime.timedelta(seconds=exec_time)))

    else:
        logger.info('Skipping, target file already exits: {path}'.format(path=output_path))

#----------------------------------------------------------------------------------------------------

def fill_holes_batch(job_list,
                     diagonal_threshold,
                     full_connectivity,
                     pixel_spacing,
                     spacing_tolerance,
                     foreground_labels=None,
                     fill_value=1,
                     copy_path=None,
                     work_path=None,
                     clear_cache=True,
                     overwrite=True):
    """
    Fill holes that have smaller than the configured diagonal in batch. The images are loaded to memory as a whole. Large images may not fit into memory.

    Args:
        job_list (list): List of job (input path, output path) tuples.
        diagonal_threshold (float): Region diagonal low threshold (micrometer).
        full_connectivity (bool): Connectivity matrix. If true edge and point neighbors will be used otherwise edge neighbors only.
        pixel_spacing (float, None): Processing pixel spacing (micrometer). If None, the lowest level is used.
        spacing_tolerance (float): Pixel spacing tolerance (percentage).
        foreground_labels (list, None): List of labels to consider as foreground, everything else is background. If empty every nonzero value is foreground.
        fill_value (int): Label value for overwriting the identified region holes.
        copy_path (str, None): Copy directory path where the images and masks are cached before processing.
        work_path (str, None): Work directory path where the output is cached first before writing it to the target path.
        clear_cache (bool): Remove the cached images.
        overwrite (bool): If true existing targets will be overwritten.

    Returns:
        (list, list): List of successfully processed items and list of failed items.
    """

    # Start time measurement.
    #
    start_time = time.time()

    # Log batch processing.
    #
    logger = logging.getLogger(name=__name__)

    logger.info('Filling holes in batch mode...')
    logger.info('Job count: {count}'.format(count=len(job_list)))

    # Go through the job list.
    #
    failed_collection = []
    successful_collection = []
    for job_index in range(len(job_list)):
        # Get one job item.
        #
        input_path, output_path = job_list[job_index]

        try:
            # Print data paths if the mode is not a single file mode.
            #
            logger.info('Processing [{index}/{count}]: {path}'.format(index=job_index+1, count=len(job_list), path=input_path))

            # Fill the holes.
            #
            fill_holes_image(image=input_path,
                             output_path=output_path,
                             diagonal_threshold=diagonal_threshold,
                             full_connectivity=full_connectivity,
                             pixel_spacing=pixel_spacing,
                             spacing_tolerance=spacing_tolerance,
                             foreground_labels=foreground_labels,
                             fill_value=fill_value,
                             copy_path=copy_path,
                             work_path=work_path,
                             clear_cache=clear_cache,
                             overwrite=overwrite)

        except Exception as exception:
            # Add case to the error collection.
            #
            failed_collection.append(output_path)
            logger.error('Error: {exception}'.format(exception=exception))

        else:
            # Save the result to the list of successful thresholds.
            #
            successful_collection.append(output_path)

    # Log execution time.
    #
    execution_time = time.time() - start_time
    logger.debug('Hole filling in batch done in {delta}'.format(delta=datetime.timedelta(seconds=execution_time)))

    # Return a list of successful filterings.
    #
    return successful_collection, failed_collection

#----------------------------------------------------------------------------------------------------

def add_region_margins_array(input_array, source_label, target_internal_label, target_external_label, margin_size):
    """
    Add margin around the regions with the given label. The function modifies the input array.

    Args:
        input_array (np.ndarray): Input array. Modified.
        source_label (int): Label to process.
        target_internal_label (int): Label of the internal margin.
        target_external_label (int): Label of the external margin.
        margin_size (int): Margin size (pixels).

    Returns:
        np.ndarray: The input array with the added margins.
    """

    # Binarize the input array.
    #
    input_array_binary = np.equal(input_array, source_label)

    # Calculate internal and external areas.
    #
    internal_area = np.logical_and(input_array_binary, np.logical_not(scipy.ndimage.binary_erosion(input=input_array_binary, iterations=margin_size)))
    external_area = np.logical_and(np.logical_not(input_array_binary), scipy.ndimage.binary_dilation(input=input_array_binary, iterations=margin_size))

    # Modify the content and write it out.
    #
    input_array[internal_area] = target_internal_label
    input_array[external_area] = target_external_label

    # Return the result image with added margins.
    #
    return input_array

#----------------------------------------------------------------------------------------------------

def add_region_margins_image(input_mask,
                             output_mask_path,
                             source_label,
                             target_internal_label,
                             target_external_label,
                             margin_size,
                             pixel_spacing,
                             spacing_tolerance,
                             copy_path=None,
                             work_path=None,
                             clear_cache=True,
                             overwrite=True):
    """
    Process a regions identified by the given label and add internal and external margin regions to the original mask item.

    Args:
        input_mask (dptimagereader.ImageReader, str): Input image object or path.
        output_mask_path (str): Output image path.
        source_label (int): Label to process.
        target_internal_label (int): Label of the internal margin.
        target_external_label (int): Label of the external margin.
        margin_size (float): Margin size (micrometer).
        pixel_spacing (float, None): Processing pixel spacing (micrometer). If None, the lowest level is used.
        spacing_tolerance (float): Pixel spacing tolerance (percentage).
        copy_path (str, None): Copy directory path where the images and masks are cached before processing.
        work_path (str, None): Work directory path where the output is cached first before writing it to the target path.
        clear_cache (bool): Remove the cached images.
        overwrite (bool): If true existing targets will be overwritten.
    """

    # Start time measurement.
    #
    start_time = time.time()

    # Log parameters.
    #
    logger = logging.getLogger(name=__name__)

    logger.info('Adding region margins...')
    logger.info('Input: {path}'.format(path=input_mask.path if isinstance(input_mask, dptimagereader.ImageReader) else input_mask))
    logger.info('Output: {path}'.format(path=output_mask_path))
    logger.info('Source label: {label}'.format(label=source_label))
    logger.info('Target internal label: {label}'.format(label=target_internal_label))
    logger.info('Target external label: {label}'.format(label=target_external_label))
    logger.info('Margin size: {size} um'.format(size=margin_size))

    if pixel_spacing is not None:
        logger.info('Pixel spacing: {spacing} um'.format(spacing=pixel_spacing))
    else:
        logger.info('Pixel spacing: at lowest level')

    # Check if target already exits.
    #
    if not os.path.isfile(output_mask_path) or overwrite:
        # Open image.
        #
        input_mask_image = input_mask if isinstance(input_mask, dptimagereader.ImageReader) else dptimagereader.ImageReader(image_path=input_mask,
                                                                                                                            spacing_tolerance=spacing_tolerance,
                                                                                                                            input_channels=None,
                                                                                                                            cache_path=copy_path)

        # Identify processing level and dilation iteration count.
        #
        processing_level = input_mask_image.level(spacing=pixel_spacing) if pixel_spacing is not None else 0
        iteration_count = round(margin_size / input_mask_image.spacings[processing_level])

        logger.debug('Identified processing level: {level}'.format(level=processing_level))

        # Read image to an array.
        #
        input_content = input_mask_image.content(spacing=input_mask_image.spacings[processing_level])
        input_content = input_content.squeeze()

        logger.info('Processing: {path}'.format(path=input_mask_image.path))

        # Add margin.
        #
        input_content = add_region_margins_array(input_array=input_content,
                                                 source_label=source_label,
                                                 target_internal_label=target_internal_label,
                                                 target_external_label=target_external_label,
                                                 margin_size=iteration_count)

        tile_size = 512
        image_writer = dptimagewriter.ImageWriter(image_path=output_mask_path,
                                                  shape=input_mask_image.shapes[processing_level],
                                                  spacing=input_mask_image.spacings[processing_level],
                                                  dtype=input_mask_image.dtype,
                                                  coding=input_mask_image.coding,
                                                  compression=None,
                                                  interpolation=None,
                                                  tile_size=tile_size,
                                                  jpeg_quality=None,
                                                  empty_value=0,
                                                  skip_empty=None,
                                                  cache_path=work_path)

        input_content = input_content[None, ...]
        image_writer.fill(content=input_content)
        image_writer.close(clear=clear_cache)

        # Close input image.
        #
        if not isinstance(input_mask, dptimagereader.ImageReader):
            input_mask_image.close(clear=clear_cache)

        # Log execution time.
        #
        execution_time = time.time() - start_time
        logger.debug('Done in {delta}'.format(delta=datetime.timedelta(seconds=execution_time)))

    else:
        logger.info('Skipping, target file already exits: {path}'.format(path=output_mask_path))

#----------------------------------------------------------------------------------------------------

def add_region_margins_batch(job_list,
                             source_label,
                             target_internal_label,
                             target_external_label,
                             margin_size,
                             pixel_spacing,
                             spacing_tolerance,
                             copy_path=None,
                             work_path=None,
                             clear_cache=True,
                             overwrite=True):
    """
    Process a regions identified by the given label and add internal and external margin regions to the original mask item.

    Args:
        job_list (list): List of job (input path, output path) tuples.
        source_label (int): Label to process.
        target_internal_label (int): Label of the internal margin.
        target_external_label (int): Label of the external margin.
        margin_size (float): Margin size (micrometer).
        pixel_spacing (float, None): Processing pixel spacing (micrometer). If None, the lowest level is used.
        spacing_tolerance (float): Pixel spacing tolerance (percentage).
        copy_path (str, None): Copy directory path where the images and masks are cached before processing.
        work_path (str, None): Work directory path where the output is cached first before writing it to the target path.
        clear_cache (bool): Remove the cached images.
        overwrite (bool): If true existing targets will be overwritten.

    Returns:
        (list, list): List of successfully processed items and list of failed items.
    """

    # Start time measurement.
    #
    start_time = time.time()

    # Log batch processing.
    #
    logger = logging.getLogger(name=__name__)
    logger.info('Adding region margins in batch mode...')
    logger.info('Job count: {count}'.format(count=len(job_list)))

    # Go through the job list.
    #
    failed_collection = []
    successful_collection = []
    for job_index in range(len(job_list)):
        # Get one job item.
        #
        input_mask_path, output_mask_path = job_list[job_index]

        try:
            # Print data paths if the mode is not a single file mode.
            #
            logger.info('Processing [{index}/{count}]: {path}'.format(index=job_index+1, count=len(job_list), path=input_mask_path))

            # Convert the annotation.
            #
            add_region_margins_image(input_mask=input_mask_path,
                                     output_mask_path=output_mask_path,
                                     source_label=source_label,
                                     target_internal_label=target_internal_label,
                                     target_external_label=target_external_label,
                                     margin_size=margin_size,
                                     pixel_spacing=pixel_spacing,
                                     spacing_tolerance=spacing_tolerance,
                                     copy_path=copy_path,
                                     work_path=work_path,
                                     clear_cache=clear_cache,
                                     overwrite=overwrite)

        except Exception as exception:
            # Add case to the error collection.
            #
            failed_collection.append(output_mask_path)
            logger.error('Error: {exception}'.format(exception=exception))

        else:
            # Save the annotation to the list of successful conversions.
            #
            successful_collection.append(output_mask_path)

    # Log execution time.
    #
    execution_time = time.time() - start_time
    logger.debug('Margin addition in batch done in {delta}'.format(delta=datetime.timedelta(seconds=execution_time)))

    # Return a list of successful conversions.
    #
    return successful_collection, failed_collection
