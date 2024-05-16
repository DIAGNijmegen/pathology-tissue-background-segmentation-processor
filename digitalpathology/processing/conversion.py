"""
This module can convert ASAP annotation descriptor XMLs to mask files and clear the parts from a mask of an image
where the patch is completely empty, normalize the label values to the given interval, calculate a classification
preview by calculating an overlay image with the original and the classification output mask, and save any level
of a multi-resolution image as a plain image.
"""

from ..errors import processingerrors as dptprocessingerrors
from ..image import annotation as dptannotation
from ..image import imagereader as dptimagereader
from ..image import imagewriter as dptimagewriter
from ..image import plainimage as dptplainimage
from ..utils import imagefile as dptimagefile

import numpy as np
import skimage.color
import logging
import datetime
import time
import os

#----------------------------------------------------------------------------------------------------

def _calculate_empty_map(image, tile_size, processing_level):
    """
    Calculates the map of non-empty tiles. If a tile contains non-zero value the value of its index at the result array is True, otherwise it is False.

    Args:
        image (dptimagereader.ImageReader): Image object to check.
        tile_size (int): Tile size.
        processing_level (int): Processing level of the image.

    Returns:
        (np.ndarray): Array of booleans indicating non completely empty patches at the given level of the input image.
    """

    # Calculate the size of the array. One bool for each image_size*image_size sized patch.
    #
    flag_array_shape = (image.shapes[processing_level][0] // tile_size + (1 if image.shapes[processing_level][0] % tile_size else 0),
                        image.shapes[processing_level][1] // tile_size + (1 if image.shapes[processing_level][1] % tile_size else 0))

    flag_array = np.zeros(shape=flag_array_shape, dtype=bool)

    # Go through the patches and mark non empty ones.
    #
    for row in range(0, flag_array_shape[0]):
        row_index = row * tile_size
        for col in range(0, flag_array_shape[1]):
            col_index = col * tile_size

            image_patch = image.read(spacing=image.spacings[processing_level], row=row_index, col=col_index, height=tile_size, width=tile_size)

            flag_array[row, col] = not np.all(np.logical_or(np.equal(image_patch, 0), np.equal(image_patch, 255)))

    # Return the per tile flag array.
    #
    return flag_array

#----------------------------------------------------------------------------------------------------

def _map_and_clear_image(mask_image, output_image, tile_size, label_map, empty_map):
    """
    Map the label values and clear the marked empty parts from the given mask image.

    Args:
        mask_image (dptimagereader.ImageReader): Mask to process.
        output_image (dptimagewriter.ImageWriter): Output mask image.
        tile_size (int): Processing tile size.
        label_map (dict): Label mapping.
        empty_map (np.ndarray, None): Map of empty patches.

    Raises:
        DigitalPathologyConfigError: Configuration errors.
        DigitalPathologyImageError: Image errors.
    """

    # Go through the patches and clear the empty ones.
    #
    if empty_map is None:
        for row in range(0, mask_image.shapes[0][0], tile_size):
            for col in range(0, mask_image.shapes[0][1], tile_size):
                image_patch = mask_image.read(spacing=mask_image.spacings[0], row=row, col=col, height=tile_size, width=tile_size)

                normalized_patch = image_patch.copy()

                for key, value in label_map.items():
                    normalized_patch[image_patch == key] = value

                output_image.write(tile=normalized_patch, row=row, col=col)
    else:
        for row in range(empty_map.shape[0]):
            row_index = row * tile_size
            for col in range(empty_map.shape[1]):
                col_index = col * tile_size

                if empty_map[row, col]:
                    image_patch = mask_image.read(spacing=mask_image.spacings[0], row=row_index, col=col_index, height=tile_size, width=tile_size)

                    normalized_patch = image_patch.copy()

                    for key, value in label_map.items():
                        normalized_patch[image_patch == key] = value

                    output_image.write(tile=normalized_patch, row=row_index, col=col_index)
                else:
                    output_image.write(tile=None, row=row_index, col=col_index)

#----------------------------------------------------------------------------------------------------

def create_annotation_mask(image,
                           annotation,
                           label_map,
                           conversion_order,
                           conversion_spacing,
                           spacing_tolerance,
                           output_path,
                           strict,
                           accept_all_empty,
                           work_path=None,
                           clear_cache=True,
                           overwrite=True):
    """
    Create mask files from annotation files.

    Args:
        image (dptimagereader.ImageReader, str): Image object or path.
        annotation (dptannotation.Annotation, str): Annotation converter object or annotation file path.
        label_map (dict): Annotation group to label value map.
        conversion_order (list): Annotation group conversion order.
        conversion_spacing (float, None): Conversion pixel spacing of the input image (micrometer). The annotation is converted on the lowest level of the image if not specified.
        spacing_tolerance (float): Pixel spacing tolerance (percentage).
        output_path (str): Result mask file path.
        strict (bool): Stop on unknown annotation groups.
        accept_all_empty (bool): Accept if all the selected groups are empty.
        work_path (str, None): Work directory path where the output is cached first before writing it to the target path.
        clear_cache (bool): Remove the cached images.
        overwrite (bool): If true existing targets will be overwritten.

    Raises:
        EmptyAnnotationGroupsError: All the selected annotation group are empty and empty groups are not allowed.
        UnknownAnnotationGroupsError: There are unknown annotation groups present and strict mode is enabled.

        DigitalPathologyConfigError: Configuration errors.
        DigitalPathologyImageError: Image errors.
    """

    # Start time measurement.
    #
    start_time = time.time()

    # Log parameters.
    #
    logger = logging.getLogger(name=__name__)

    logger.info('Converting annotation to mask...')
    logger.info('Annotation: {path}'.format(path=annotation.path if isinstance(annotation, dptannotation.Annotation) else annotation))
    logger.info('Image: {path}'.format(path=image.path if isinstance(image, dptimagereader.ImageReader) else image))
    logger.info('Mask: {path}'.format(path=output_path))
    logger.info('Label map: {map}'.format(map=label_map))
    logger.info('Conversion order: {order}'.format(order=conversion_order))
    logger.info('Conversion spacing: {spacing} um'.format(spacing=conversion_spacing))

    # Check if target already exits.
    #
    if not os.path.isfile(output_path) or overwrite:
        # Calculate work path.
        #
        work_output_path = os.path.join(work_path, os.path.basename(output_path)) if work_path else output_path

        # Create work directory structure if necessary.
        #
        if work_path:
            os.makedirs(os.path.dirname(work_path), exist_ok=True)

        # Open image.
        #
        template_image = image if isinstance(image, dptimagereader.ImageReader) else dptimagereader.ImageReader(image_path=image,
                                                                                                                spacing_tolerance=spacing_tolerance,
                                                                                                                input_channels=None,
                                                                                                                cache_path=None)

        used_conversion_spacing = conversion_spacing if conversion_spacing is not None else template_image.spacings[0]
        conversion_level = template_image.level(spacing=used_conversion_spacing)

        # Load the annotations from file.
        #
        if isinstance(annotation, dptannotation.Annotation):
            annotation_conv = annotation
        else:
            annotation_conv = dptannotation.Annotation()
            annotation_conv.open(annotation_path=annotation, spacing=template_image.spacings[0])

        # Check if there is at least one annotation.
        #
        if not accept_all_empty:
            annotation_counts = annotation_conv.counts()
            if label_map is not None and sum(annotation_counts[group_name] for group_name in annotation_counts if group_name in label_map) == 0 or label_map is None and not annotation_counts:
                raise dptprocessingerrors.EmptyAnnotationGroupsError(list(label_map.keys()))

        # Check if there is an unknown group name.
        #
        if strict:
            unknown_group_names = set(annotation_conv.groups.keys()) - set(label_map.keys())
            if unknown_group_names:
                # Raise error on unknown group.
                #
                raise dptprocessingerrors.UnknownAnnotationGroupsError(unknown_group_names)

        logger.info('Converting: {path}'.format(path=annotation_conv.path))

        # Convert the annotations to mask.
        #
        annotation_conv.convert(image_path=work_output_path,
                                shape=template_image.shapes[conversion_level],
                                spacing=template_image.spacings[conversion_level],
                                label_map=label_map,
                                conversion_order=conversion_order)

        # Close the template image.
        #
        if not isinstance(image, dptimagereader.ImageReader):
            template_image.close(clear=True)

        # Copy the result to the target location.
        #
        if work_path:
            dptimagefile.relocate_image(source_path=work_output_path, target_path=output_path, move=clear_cache, overwrite=overwrite)

        # Log execution time.
        #
        execution_time = time.time() - start_time
        logger.debug('Done in {delta}'.format(delta=datetime.timedelta(seconds=execution_time)))

    else:
        logger.info('Skipping, target file already exits: {path}'.format(path=output_path))

#----------------------------------------------------------------------------------------------------

def create_annotation_mask_batch(job_list,
                                 label_map,
                                 conversion_order,
                                 conversion_spacing,
                                 spacing_tolerance,
                                 strict,
                                 accept_all_empty,
                                 work_path=None,
                                 clear_cache=True,
                                 overwrite=True):
    """
    Go through all the listed jobs and create mask a file from each annotation file.

    Args:
        job_list (list): List of job (image path, annotation path, mask path) tuples.
        label_map (dict): Annotation group to label value map.
        conversion_order (list): Annotation group conversion order.
        conversion_spacing (float, None): Conversion pixel spacing of the input image (micrometer). The annotation is converted on the lowest level of the image if not specified.
        spacing_tolerance (float): Pixel spacing tolerance (percentage).
        strict (bool): Stop on unknown annotation groups.
        accept_all_empty (bool): Accept if all the selected groups are empty.
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

    logger.info('Converting annotations to masks in batch mode...')
    logger.info('Job count: {count}'.format(count=len(job_list)))

    # Go through the job list.
    #
    failed_collection = []
    successful_collection = []
    for job_index in range(len(job_list)):
        # Get one job item.
        #
        image_path, annotation_path, output_path = job_list[job_index]

        try:
            # Print data paths if the mode is not a single file mode.
            #
            logger.info('Processing [{index}/{count}]: {path}'.format(index=job_index+1, count=len(job_list), path=annotation_path))

            # Convert the annotation.
            #
            create_annotation_mask(image=image_path,
                                   annotation=annotation_path,
                                   label_map=label_map,
                                   conversion_order=conversion_order,
                                   conversion_spacing=conversion_spacing,
                                   spacing_tolerance=spacing_tolerance,
                                   output_path=output_path,
                                   strict=strict,
                                   accept_all_empty=accept_all_empty,
                                   work_path=work_path,
                                   clear_cache=clear_cache,
                                   overwrite=overwrite)

        except Exception as exception:
            # Add case to the error collection.
            #
            failed_collection.append(output_path)
            logger.error('Error: {exception}'.format(exception=exception))

        else:
            # Save the annotation to the list of successful conversions.
            #
            successful_collection.append(output_path)

    # Log execution time.
    #
    execution_time = time.time() - start_time
    logger.debug('Annotation conversion batch done in {delta}'.format(delta=datetime.timedelta(seconds=execution_time)))

    # Return a list of successful conversions.
    #
    return successful_collection, failed_collection

#----------------------------------------------------------------------------------------------------

def normalize_mask_image(mask, tile, output_path, label_map, empty_label, copy_path=None, work_path=None, clear_cache=True, overwrite=True):
    """
    Normalize mask image. First identify and fill the patches on the mask label with the empty value where the corresponding image is empty, then rearrange
    the label values in the mask according to the label value mapping.

    Args:
        mask (dptimagereader.ImageReader, str): Mask image object or path.
        tile (dptimagereader.ImageReader, str, None): Input image object or path. The image that will be used for empty part recognition.
        output_path (str): Output image path. The zoomed image.
        label_map (dict): Label mapping.
        empty_label (int): Empty patch label value.
        copy_path (str, None): Copy directory path where the images and masks are cached before processing.
        work_path (str, None): Work directory path where the output is cached first before writing it to the target path.
        clear_cache (bool): Remove the cached images.
        overwrite (bool): If true existing targets will be overwritten.

    Raises:
        NoMatchingLevelInTileImageError: Cannot find matching level for mask in tile image.

        DigitalPathologyConfigError: Configuration errors.
        DigitalPathologyImageError: Image errors.
    """

    # Start time measurement.
    #
    start_time = time.time()

    # Log parameters.
    #
    logger = logging.getLogger(name=__name__)

    logger.info('Normalizing mask image...')
    logger.info('Mask: {path}'.format(path=mask.path if isinstance(mask, dptimagereader.ImageReader) else mask))
    logger.info('Template: {path}'.format(path=tile.path if isinstance(tile, dptimagereader.ImageReader) else tile))
    logger.info('Normalized: {path}'.format(path=output_path))
    logger.info('Label map: {map}'.format(map=label_map))
    logger.info('Empty label: {label}'.format(label=empty_label))

    # Check if the target image already exits.
    #
    if not os.path.isfile(output_path) or overwrite:
        # Open images.
        #
        if tile is not None:
            tile_image = tile if isinstance(tile, dptimagereader.ImageReader) else dptimagereader.ImageReader(image_path=tile, spacing_tolerance=0.25, input_channels=None, cache_path=copy_path)
        else:
            tile_image = None

        mask_image = mask if isinstance(mask, dptimagereader.ImageReader) else dptimagereader.ImageReader(image_path=mask, spacing_tolerance=0.25, input_channels=None, cache_path=copy_path)

        if tile_image:
            if any(level_spacing is None for level_spacing in tile_image.spacings):
                tile_image.correct(spacing=1.0, level=0)

        if any(level_spacing is None for level_spacing in mask_image.spacings):
            mask_image.correct(spacing=1.0, level=0)

        # Basic configuration.
        #
        tile_size = 512  # Patch size for image reading and writing.

        # Calculate empty map.
        #
        if tile_image:
            # Find matching level with the tile image.
            #
            mask_dimensions = mask_image.shapes[0]
            level = None
            for level_index in range(len(tile_image.shapes)):
                if mask_dimensions == tile_image.shapes[level_index]:
                    level = level_index
                    break

            if level is None:
                raise dptprocessingerrors.NoMatchingLevelInTileImageError(tile_image.path, mask_image.path)

            logger.info('Identifying empty tiles: {path}'.format(path=tile_image.path))

            empty_map = _calculate_empty_map(image=tile_image, tile_size=tile_size, processing_level=level)
        else:
            empty_map = None

        # Clean empty parts.
        #
        logger.info('Normalizing mask: {path}'.format(path=output_path))

        # Configure the image writer.
        #
        output_image = dptimagewriter.ImageWriter(image_path=output_path,
                                                  shape=mask_image.shapes[0],
                                                  spacing=mask_image.spacings[0],
                                                  dtype=mask_image.dtype,
                                                  coding=mask_image.coding,
                                                  compression=None,
                                                  interpolation=None,
                                                  tile_size=tile_size,
                                                  jpeg_quality=None,
                                                  empty_value=empty_label,
                                                  skip_empty=None,
                                                  cache_path=work_path)

        _map_and_clear_image(mask_image=mask_image, output_image=output_image, tile_size=tile_size, label_map=label_map, empty_map=empty_map)

        # Finalize result.
        #
        output_image.close(clear=clear_cache)

        # Close images.
        #
        if not isinstance(mask, dptimagereader.ImageReader):
            mask_image.close(clear=clear_cache)

        if tile_image:
            if not isinstance(tile, dptimagereader.ImageReader):
                tile_image.close(clear=clear_cache)

        # Log execution time.
        #
        execution_time = time.time() - start_time
        logger.debug('Done in {delta}'.format(delta=datetime.timedelta(seconds=execution_time)))

    else:
        logger.info('Skipping, target file already exits: {path}'.format(path=output_path))

#----------------------------------------------------------------------------------------------------

def normalize_mask_image_batch(job_list, label_map, empty_label, copy_path=None, work_path=None, clear_cache=True, overwrite=True):
    """
    Normalize mask images. Tile path will be used to identify empty tiles.

    Args:
        job_list (list): List of job (mask path, tile path, output path) tuples.
        label_map (dict): Label mapping.
        empty_label (int): Empty patch label value.
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

    logger.info('Normalizing mask images in batch mode...')
    logger.info('Job count: {count}'.format(count=len(job_list)))

    # Go through the job list.
    #
    failed_collection = []
    successful_collection = []
    for job_index in range(len(job_list)):
        # Get one job item.
        #
        mask_path, tile_path, output_path = job_list[job_index]

        try:
            # Print data paths if the mode is not a single file mode.
            #
            logger.info('Processing [{index}/{count}]: {path}'.format(index=job_index+1, count=len(job_list), path=mask_path))

            # Normalize the mask image.
            #
            normalize_mask_image(mask=clear_cache,
                                 tile=tile_path,
                                 output_path=output_path,
                                 label_map=label_map,
                                 empty_label=empty_label,
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
            # Save the mask to the list of successful conversions.
            #
            successful_collection.append(output_path)

    # Log execution time.
    #
    execution_time = time.time() - start_time
    logger.debug('Mask image normalization batch done in {delta}'.format(delta=datetime.timedelta(seconds=execution_time)))

    # Return a list of successful normalizations.
    #
    return successful_collection, failed_collection

#----------------------------------------------------------------------------------------------------

def create_mask_annotation(mask, annotation, label_map, conversion_spacing, target_spacing, spacing_tolerance, keep_singles, rdp_epsilon, overwrite=True):
    """
    Create annotation by outlining regions in the mask image.

    Args:
        mask (dptimagereader.ImageReader, str): Source mask image object or path.
        annotation (dptannotation.Annotation, str): Target annotation object or path.
        label_map (dict, None): Annotation group to label value list map.
        conversion_spacing (float): Conversion pixel spacing of the input mask (micrometer).
        target_spacing (float, None): Target pixel spacing pixel spacing of the annotation (micrometer). Same as the conversion spacing if None.
        spacing_tolerance (float): Pixel spacing tolerance (percentage).
        keep_singles (bool): Keep single point annotations.
        rdp_epsilon (float): RDP epsilon.
        overwrite (bool): If true existing targets will be overwritten.

    Raises:
        DigitalPathologyConfigError: Configuration errors.
        DigitalPathologyImageError: Image errors.
    """

    # Start time measurement.
    #
    start_time = time.time()

    # Get the paths.
    #
    mask_path = mask.path if isinstance(mask, dptimagereader.ImageReader) else mask
    annotation_path = annotation.path if isinstance(annotation, dptannotation.Annotation) else annotation

    # Log parameters.
    #
    logger = logging.getLogger(name=__name__)

    logger.info('Converting mask to annotation...')
    logger.info('Mask: {path}'.format(path=mask_path))
    logger.info('Annotation: {path}'.format(path=annotation_path))
    logger.info('Label map: {map}'.format(map=label_map))
    logger.info('Conversion spacing: {spacing} um'.format(spacing=conversion_spacing))
    logger.info('Target spacing: {spacing} um'.format(spacing=target_spacing))
    logger.info('Keep single point annotations: {flag}'.format(flag=keep_singles))
    logger.info('Ramer-Douglas-Peucker epsilon: {value}'.format(value=rdp_epsilon))

    # Check if target already exits.
    #
    if isinstance(annotation, dptannotation.Annotation) or not os.path.isfile(annotation_path) or overwrite:
        # Open image and target annotation.
        #
        input_mask = mask if isinstance(mask, dptimagereader.ImageReader) else dptimagereader.ImageReader(image_path=mask,
                                                                                                          spacing_tolerance=spacing_tolerance,
                                                                                                          input_channels=None,
                                                                                                          cache_path=None)

        output_annotation = annotation if isinstance(annotation, dptannotation.Annotation) else dptannotation.Annotation()

        # Refine the spacings.
        #
        input_pixel_spacing = input_mask.refine(spacing=conversion_spacing)
        if target_spacing is None:
            output_pixel_spacing = input_pixel_spacing
        else:
            output_pixel_spacing = input_mask.refine(spacing=target_spacing) if input_mask.test(spacing=target_spacing) else target_spacing

        # Convert the mask to annotation.
        #
        output_annotation.outline(image=input_mask, spacing=input_pixel_spacing, spacing_tolerance=spacing_tolerance, label_map=label_map, single_points=keep_singles, rdp_epsilon=rdp_epsilon)

        # Save the annotations to file if the parameter was a file path.
        #
        if not isinstance(annotation, dptannotation.Annotation):
            output_annotation.save(annotation_path=annotation, spacing=output_pixel_spacing)

        # Close the input mask.
        #
        if not isinstance(mask, dptimagereader.ImageReader):
            input_mask.close()

        # Log execution time.
        #
        execution_time = time.time() - start_time
        logger.debug('Done in {delta}'.format(delta=datetime.timedelta(seconds=execution_time)))

    else:
        logger.info('Skipping, target file already exits: {path}'.format(path=annotation_path))

#----------------------------------------------------------------------------------------------------

def create_mask_annotation_batch(job_list, label_map, conversion_spacing, target_spacing, spacing_tolerance, keep_singles, rdp_epsilon, overwrite=True):
    """
    Go through all the listed jobs and create annotation outline each mask file.

    Args:
        job_list (list): List of job (mask path, annotation path) tuples.
        label_map (dict): Annotation group to label value list map.
        conversion_spacing (float): Conversion pixel spacing of the input mask (micrometer).
        target_spacing (float, None): Target pixel spacing pixel spacing of the annotation (micrometer). Same as the conversion spacing if None.
        spacing_tolerance (float): Pixel spacing tolerance (percentage).
        keep_singles (bool): Keep single point annotations.
        rdp_epsilon (float): RDP epsilon.
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

    logger.info('Converting masks to annotations in batch mode...')
    logger.info('Job count: {count}'.format(count=len(job_list)))

    # Go through the job list.
    #
    failed_collection = []
    successful_collection = []
    for job_index in range(len(job_list)):
        # Get one job item.
        #
        mask_path, annotation_path = job_list[job_index]

        try:
            # Print data paths if the mode is not a single file mode.
            #
            logger.info('Processing [{index}/{count}]: {path}'.format(index=job_index+1, count=len(job_list), path=mask_path))

            # Convert the annotation.
            #
            create_mask_annotation(mask=mask_path,
                                   annotation=annotation_path,
                                   label_map=label_map,
                                   conversion_spacing=conversion_spacing,
                                   target_spacing=target_spacing,
                                   spacing_tolerance=spacing_tolerance,
                                   keep_singles=keep_singles,
                                   rdp_epsilon=rdp_epsilon,
                                   overwrite=overwrite)

        except Exception as exception:
            # Add case to the error collection.
            #
            failed_collection.append(annotation_path)
            logger.error('Error: {exception}'.format(exception=exception))

        else:
            # Save the annotation to the list of successful conversions.
            #
            successful_collection.append(annotation_path)

    # Log execution time.
    #
    execution_time = time.time() - start_time
    logger.debug('Mask conversion batch done in {delta}'.format(delta=datetime.timedelta(seconds=execution_time)))

    # Return a list of successful conversions.
    #
    return successful_collection, failed_collection

#----------------------------------------------------------------------------------------------------

def calculate_preview(image, mask, preview_path, level, pixel_spacing, spacing_tolerance, alpha, palette=None, copy_path=None, clear_cache=True, overwrite=True):
    """
    Create a preview image from the given image and mask on the given level by overlaying the original RGB image with a false colored mask label image (typically the output of the network inference).
    The palette of the coloring can be set. By default the mask value 0 is transparent, 1 is red, 2 is green, 3 is blue, 4 is yellow, etc. This function can be used to generate preview for the
    output of a network on a high image level to quickly review the results.

    Args:
        image (dptimagereader.ImageReader, str): Input image object or path.
        mask (dptimagereader.ImageReader, str): Input mask object or path.
        preview_path (str): Path of the result preview image file.
        level (int, None): Processing level of the image. Either this or the pixel_spacing parameter is required.
        pixel_spacing (float, None): Pixel spacing of the image to process. (micrometer). Either this or the level parameter is required.
        spacing_tolerance (float, None): Pixel spacing tolerance (percentage).
        alpha (float): Overlay alpha value for preview generation.
        palette (list, None): List of palette colors. Each color must be a tuple with 3 elements from the [0, 255] range.
        copy_path (str, None): Copy directory path where the image and mask images are cached before processing.
        clear_cache (bool): Remove the cached images.
        overwrite (bool): If true existing targets will be overwritten.

    Raises:
        InvalidPixelSpacingValueError: Both image level and pixel spacing are None.
        InvalidColorPaletteError: Invalid palette for mask image coloring.
        NoMatchingLevelInMaskImageError: No matching level found in the mask image for the RGB image.

        DigitalPathologyConfigError: Configuration errors.
        DigitalPathologyImageError: Image errors.
    """

    # Start time measurement.
    #
    start_time = time.time()

    # Check level and pixel spacing. One must be valid.
    #
    if level is None and pixel_spacing is None:
        raise dptprocessingerrors.InvalidPixelSpacingValueError()

    # Check the palette.
    #
    if palette:
        if len(palette) < 1 or any(len(color) != 3 or any(color_item < 0 or 255 < color_item for color_item in color) for color in palette):
            raise dptprocessingerrors.InvalidColorPaletteError(palette)

    # Log parameters.
    #
    logger = logging.getLogger(name=__name__)

    logger.info('Calculating preview...')
    logger.info('Image: {path}'.format(path=image.path if isinstance(image, dptimagereader.ImageReader) else image))
    logger.info('Mask: {path}'.format(path=mask.path if isinstance(mask, dptimagereader.ImageReader) else mask))
    logger.info('Preview: {path}'.format(path=preview_path))

    if level is None:
        logger.info('Processing spacing: {spacing} um'.format(spacing=pixel_spacing))
    else:
        logger.info('Processing level: {level}'.format(level=level))

    # Check if the target image already exits.
    #
    if not os.path.isfile(preview_path) or overwrite:
        # Open input image and mask image.
        #
        input_image = image if isinstance(image, dptimagereader.ImageReader) else dptimagereader.ImageReader(image_path=image,
                                                                                                             spacing_tolerance=spacing_tolerance,
                                                                                                             input_channels=None,
                                                                                                             cache_path=copy_path)

        input_mask = mask if isinstance(mask, dptimagereader.ImageReader) else dptimagereader.ImageReader(image_path=mask,
                                                                                                          spacing_tolerance=spacing_tolerance,
                                                                                                          input_channels=None,
                                                                                                          cache_path=copy_path)

        # Calculate source level and target spacing and add missing spacing information.
        #
        processing_level = input_image.level(spacing=pixel_spacing) if level is None else level

        if level is None:
            logger.debug('Identified level: {level}'.format(level=processing_level))

        if any(level_spacing is None for level_spacing in input_mask.spacings):
            input_mask.correct(spacing=1.0, level=0)

        # Find matching level for the mask with the image.
        #
        mask_level = None
        for level_index in range(len(input_mask.shapes)):
            if input_image.shapes[processing_level] == input_mask.shapes[level_index]:
                mask_level = level_index
                break

        if mask_level is None:
            raise dptprocessingerrors.NoMatchingLevelInMaskImageError(input_mask.path, input_image.path, processing_level)

        logger.info('Calculating preview: {path}'.format(path=preview_path))

        # Construct default palette for preview: red, green, blue, orange, magenta, purple, yellow, pink, teal
        #
        if palette:
            preview_palette = [[palette_item[0] / 255.0, palette_item[1] / 255.0, palette_item[2] / 255.0] for palette_item in palette]
        else:
            preview_palette = [[0.0, 0.0, 0.0],  # black (background)
                               [1.0, 0.0, 0.0],  # red
                               [0.0, 1.0, 0.0],  # green
                               [0.0, 0.0, 1.0],  # blue
                               [1.0, 0.5, 0.0],  # orange
                               [1.0, 0.0, 0.5],  # magenta
                               [0.5, 0.0, 1.0],  # purple
                               [1.0, 1.0, 0.0],  # yellow
                               [1.0, 0.5, 1.0],  # pink
                               [0.0, 1.0, 1.0],  # teal
                               [0.5, 0.0, 0.0]]  # brown

        # Convert the palette to numpy array for coloring.
        #
        palette_array = np.asarray(preview_palette, dtype=np.float32)

        # Load the images to numpy arrays.
        #
        image_array = input_image.content(spacing=input_image.spacings[processing_level])
        image_array = image_array.transpose((1, 2, 0))

        mask_array = input_mask.content(spacing=input_mask.spacings[mask_level])
        mask_array = mask_array.transpose((1, 2, 0)).squeeze()

        # Create triple channel alpha array.
        #
        alpha_array = np.full(shape=mask_array.shape, fill_value=alpha, dtype=np.float)
        alpha_array[mask_array == 0] = 0.0
        alpha_array = alpha_array[:, :, None]
        alpha_array = np.repeat(a=alpha_array, repeats=3, axis=2)

        # Remove color from masked areas.
        #
        image_array_float = image_array / 255.0

        image_array_gray = skimage.color.rgb2gray(rgb=image_array)
        image_array_gray = image_array_gray[:, :, None]
        image_array_gray = np.repeat(a=image_array_gray, repeats=3, axis=2)

        gray_out_flags = np.greater(alpha_array, 0.0)
        image_array_float[gray_out_flags] = image_array_gray[gray_out_flags]

        # Compose overlay and save image.
        #
        mask_rgb_array_float = palette_array[mask_array]
        image_overlaid_array = image_array_float * (1.0 - alpha_array) + mask_rgb_array_float * alpha_array

        image_overlaid_array *= 255.0
        image_overlaid_array = image_overlaid_array.astype(dtype=np.uint8)

        preview_image = dptplainimage.PlainImage()
        preview_image.fill(content=image_overlaid_array.transpose((2, 0, 1)))
        preview_image.write(image_path=preview_path)

        # Processing finished. Close the images.
        #
        if not isinstance(mask, dptimagereader.ImageReader):
            input_mask.close(clear=clear_cache)

        if not isinstance(image, dptimagereader.ImageReader):
            input_image.close(clear=clear_cache)

        # Log execution time.
        #
        execution_time = time.time() - start_time
        logger.debug('Done in {delta}'.format(delta=datetime.timedelta(seconds=execution_time)))

    else:
        logger.info('Skipping, target file already exits: {path}'.format(path=preview_path))

#----------------------------------------------------------------------------------------------------

def calculate_preview_batch(job_list, level, pixel_spacing, spacing_tolerance, alpha, palette=None, copy_path=None, clear_cache=True, overwrite=True):
    """
    Create image and mask overlay previews.

    Args:
        job_list (list): List of job (image path, mask path, preview path) tuples.
        level (int, None): Processing level of the image. Either this or the pixel_spacing parameter is required.
        pixel_spacing (float, None): Pixel spacing of the image to process. (micrometer). Either this or the level parameter is required.
        spacing_tolerance (float, None): Pixel spacing tolerance (percentage).
        alpha (float): Overlay alpha value for preview generation.
        palette (list, None): List of palette colors. Each color must be a tuple with 3 elements from the [0, 255] range.
        copy_path (str, None): Copy directory path where the image and mask images are cached before processing.
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

    logger.info('Calculating previews in batch mode...')
    logger.info('Job count: {count}'.format(count=len(job_list)))

    # Go through the job list.
    #
    failed_collection = []
    successful_collection = []
    for job_index in range(len(job_list)):
        # Get one job item.
        #
        image_path, mask_path, preview_path = job_list[job_index]

        try:
            # Print data paths if the mode is not a single file mode.
            #
            logger.info('Processing [{index}/{count}]: {path}'.format(index=job_index+1, count=len(job_list), path=preview_path))

            # Calculate the preview.
            #
            calculate_preview(image=image_path,
                              mask=mask_path,
                              preview_path=preview_path,
                              level=level,
                              pixel_spacing=pixel_spacing,
                              spacing_tolerance=spacing_tolerance,
                              alpha=alpha,
                              palette=palette,
                              copy_path=copy_path,
                              clear_cache=clear_cache,
                              overwrite=overwrite)

        except Exception as exception:
            # Add case to the error collection.
            #
            failed_collection.append(preview_path)
            logger.error('Error: {exception}'.format(exception=exception))

        else:
            # Save the image to the list of successful calculations.
            #
            successful_collection.append(preview_path)

    # Log execution time.
    #
    execution_time = time.time() - start_time
    logger.debug('Preview calculation batch done in {delta}'.format(delta=datetime.timedelta(seconds=execution_time)))

    # Return a list of successful calculations.
    #
    return successful_collection, failed_collection

#----------------------------------------------------------------------------------------------------

def save_mrimage_as_image(image, output_path, level, pixel_spacing, spacing_tolerance, multiplier=1, overwrite=True):
    """
    Save image at the required level.

    Args:
        image (dptimagereader.ImageReader, str): Input image object or path.
        output_path (str): Output image path.
        level (int, None): Processing level. Either this or the pixel_spacing parameter is required.
        pixel_spacing (float, None): Processing pixel spacing (micrometer). Either this or the level parameter is required.
        spacing_tolerance (float, None): Pixel spacing tolerance (percentage).
        multiplier (int): Multiplier to apply on the image. Might be useful in case of label images.
        overwrite (bool): If true existing targets will be overwritten.

    Raises:
        InvalidPixelSpacingValueError: Both image level and pixel spacing are None.

        DigitalPathologyConfigError: Configuration errors.
        DigitalPathologyImageError: Image errors.
    """

    # Start time measurement.
    #
    start_time = time.time()

    # Check level and pixel spacing. One must be valid.
    #
    if level is None and pixel_spacing is None:
        raise dptprocessingerrors.InvalidPixelSpacingValueError()

    # Log paths.
    #
    logger = logging.getLogger(name=__name__)

    logger.info('Saving multiresolution image as plain image...')
    logger.info('Multiresolution image: {path}'.format(path=image.path if isinstance(image, dptimagereader.ImageReader) else image))
    logger.info('Plain image: {path}'.format(path=output_path))
    logger.info('Multiplier: {multiplier}'.format(multiplier=multiplier))

    if level is None:
        logger.info('Processing spacing: {spacing} um'.format(spacing=pixel_spacing))
    else:
        logger.info('Processing level: {level}'.format(level=level))

    # Check if the target image already exits.
    #
    if not os.path.isfile(output_path) or overwrite:
        # Open images.
        #
        input_image = image if isinstance(image, dptimagereader.ImageReader) else dptimagereader.ImageReader(image_path=image,
                                                                                                             spacing_tolerance=spacing_tolerance,
                                                                                                             input_channels=None,
                                                                                                             cache_path=None)

        # Calculate processing level and add missing spacing information.
        #
        processing_level = input_image.level(spacing=pixel_spacing) if level is None else level

        if level is None:
            logger.debug('Identified level: {level}'.format(level=processing_level))

        if any(level_spacing is None for level_spacing in input_image.spacings):
            input_image.correct(spacing=1.0, level=0)

        # Read image.
        #
        image_array = input_image.content(spacing=input_image.spacings[processing_level])

        if multiplier != 1:
            image_array *= multiplier

        # Configure the image writer.
        #
        single_image = dptplainimage.PlainImage()
        single_image.fill(content=image_array)
        single_image.write(image_path=output_path)

        # Close the input image.
        #
        if not isinstance(image, dptimagereader.ImageReader):
            input_image.close(clear=True)

        # Log execution time.
        #
        execution_time = time.time() - start_time
        logger.debug('Done in {delta}'.format(delta=datetime.timedelta(seconds=execution_time)))

    else:
        logger.info('Skipping, target file already exits: {path}'.format(path=output_path))

#----------------------------------------------------------------------------------------------------

def save_mrimage_as_image_batch(job_list, level, pixel_spacing, spacing_tolerance, multiplier=1, overwrite=True):
    """
    Save image at the required level.

    Args:
        job_list (list): List of job (input path, output path) tuples.
        level (int, None): Processing level. Either this or the pixel_spacing parameter is required.
        pixel_spacing (float, None): Processing pixel spacing (micrometer). Either this or the level parameter is required.
        spacing_tolerance (float, None): Pixel spacing tolerance (percentage).
        multiplier (int): Multiplier to apply on the image. Might be useful in case of label images.
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

    logger.info('Saving multiresolution images as plain images in batch mode...')
    logger.info('Job count: {count}'.format(count=len(job_list)))

    # Go through the job list.
    #
    failed_collection = []
    successful_collection = []
    for job_index in range(len(job_list)):
        # Get one job item.
        #
        mrimage_path, output_path = job_list[job_index]

        try:
            # Print data paths if the mode is not a single file mode.
            #
            logger.info('Processing [{index}/{count}]: {path}'.format(index=job_index+1, count=len(job_list), path=output_path))

            # Execute the conversion.
            #
            save_mrimage_as_image(image=mrimage_path,
                                  output_path=output_path,
                                  level=level,
                                  pixel_spacing=pixel_spacing,
                                  spacing_tolerance=spacing_tolerance,
                                  multiplier=multiplier,
                                  overwrite=overwrite)

        except Exception as exception:
            # Add case to the error collection.
            #
            failed_collection.append(output_path)
            logger.error('Error: {exception}'.format(exception=exception))

        else:
            # Save the result to the list of successful conversions.
            #
            successful_collection.append(output_path)

    # Log execution time.
    #
    execution_time = time.time() - start_time
    logger.debug('Image saving batch done in {delta}'.format(delta=datetime.timedelta(seconds=execution_time)))

    # Return a list of successful conversions.
    #
    return successful_collection, failed_collection
