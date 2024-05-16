"""
This module can zoom a given image to the size of the given template image on the desired level or save image at a different level.
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

def zoom_image(image, output_path, zoom, level, pixel_spacing, spacing_tolerance, round_shape=True, interpolation_order=0, jpeg_quality=None, overwrite=True):
    """
    Zoom the given input image with the given zoom factor or to the given shape.

    Args:
        image (dptimagereader.ImageReader, str): Input image object or path. The image that will be zoomed.
        output_path (str): Output image path. The zoomed image.
        zoom (float, tuple): Zooming factor if a single value, target shape if a tuple.
        level (int, None): Level of the input image to use. Either this or the input_pixel_spacing parameter is required.
        pixel_spacing (float, None): Pixel spacing of the input image to use. (micrometer). Either this or the input_level parameter is required.
        spacing_tolerance (float): Pixel spacing tolerance (percentage).
        round_shape (bool): If true non-integral output shape dimensions will be rounded to the nearest integer, otherwise truncated.
        interpolation_order (int): Interpolation order for zooming from the [0, 5] interval.
        jpeg_quality (int, None): JPEG quality (1-100) when using JPEG as compression method. If not set, the default of ImageWriter is used.
        overwrite (bool): If true existing targets will be overwritten.

    Raises:
        InvalidPixelSpacingValueError: Both image level and pixel spacing are None.
        InvalidZoomFactorError: Non-positive zoom factor.
        InvalidZoomTargetShapeError: The target shape to zoom to is invalid.
        InvalidInterpolationOrderError: Interpolation order out of [0, 5] bounds.
        AsymmetricZoomFactorError: Asymmetric zoom values.

        DigitalPathologyConfigError: Configuration errors.
        DigitalPathologyImageError: Image errors.
    """

    # Start time measurement.
    #
    start_time = time.time()

    # Log parameters.
    #
    logger = logging.getLogger(name=__name__)

    logger.info('Zooming image...')
    logger.info('Input image: {path}'.format(path=image.path if isinstance(image, dptimagereader.ImageReader) else image))
    logger.info('Result image: {path}'.format(path=output_path))

    if level is None:
        logger.info('Processing spacing: {spacing} um'.format(spacing=pixel_spacing))
    else:
        logger.info('Processing level: {level}'.format(level=level))

    if type(zoom) is float:
        logger.info('Zoom factor: {factor}'.format(factor=zoom))
    else:
        logger.info('Zoom target shape: {shape}'.format(shape=zoom))

    logger.info('Interpolation order: {order}'.format(order=interpolation_order))

    # Check level and pixel spacing. One must be valid.
    #
    if level is None and pixel_spacing is None:
        raise dptprocessingerrors.InvalidPixelSpacingValueError()

    # Check zoom factor.
    #
    if type(zoom) is float:
        if zoom <= 0.0:
            raise dptprocessingerrors.InvalidZoomFactorError(zoom)
    else:
        if len(zoom) != 2 or zoom[0] <= 0 or zoom[1] <= 0:
            raise dptprocessingerrors.InvalidZoomTargetShapeError(zoom)

    # Check interpolation order.
    #
    if interpolation_order < 0 or 5 < interpolation_order:
        raise dptprocessingerrors.InvalidInterpolationOrderError(interpolation_order)

    # Check if the target image already exits.
    #
    if not os.path.isfile(output_path) or overwrite:
        # Open images.
        #
        input_image = image if isinstance(image, dptimagereader.ImageReader) else dptimagereader.ImageReader(image_path=image,
                                                                                                             spacing_tolerance=spacing_tolerance,
                                                                                                             input_channels=None,
                                                                                                             cache_path=None)

        # Calculate source level and target spacing and add missing spacing information.
        #
        processing_level = input_image.level(spacing=pixel_spacing) if level is None else level

        if level is None:
            logger.debug('Identified level: {level}'.format(level=processing_level))

        if any(level_spacing is None for level_spacing in input_image.spacings):
            input_image.correct(spacing=1.0, level=0)

        # Calculate the zoom factor.
        #
        if type(zoom) is float:
            zoom_factor = zoom
        else:
            # Calculate the zoom factors.
            #
            row_zoom_factor = zoom[0] / input_image.shapes[processing_level][0]
            col_zoom_factor = zoom[1] / input_image.shapes[processing_level][1]

            # Check if zoom is symmetric.
            #
            if not math.isclose(row_zoom_factor, col_zoom_factor, rel_tol=0.01):
                raise dptprocessingerrors.AsymmetricZoomFactorError(row_zoom_factor, col_zoom_factor)

            zoom_factor = row_zoom_factor
            logger.info('Calculated zoom factor: {factor}'.format(factor=zoom_factor))

        # Configure the image writer.
        #
        tile_size = 512
        zoomed_shape = (input_image.shapes[processing_level][0] * zoom_factor, input_image.shapes[processing_level][1] * zoom_factor)
        target_spacing = input_image.spacings[processing_level] / zoom_factor

        if type(zoom) is tuple:
            target_shape = zoom
        else:
            target_shape = (round(zoomed_shape[0]), round(zoomed_shape[1])) if round_shape else (int(zoomed_shape[0]), int(zoomed_shape[1]))
            logger.info('Target zoom shape: {shape}'.format(shape=target_shape))

        image_writer = dptimagewriter.ImageWriter(image_path=output_path,
                                                  shape=target_shape,
                                                  spacing=target_spacing,
                                                  dtype=input_image.dtype,
                                                  coding=input_image.coding,
                                                  compression=None,
                                                  interpolation=None,
                                                  tile_size=tile_size,
                                                  jpeg_quality=jpeg_quality,
                                                  empty_value=0,
                                                  skip_empty=None,
                                                  cache_path=None)

        # There could be a small difference between the row and col zooming. That could yield different target and zoom output size.
        #
        zoom_output_width = round(zoomed_shape[1])

        # Process the image by rows to prevent interpolation artifacts.
        #
        target_margin_size = 8
        use_pre_filter = 0 < interpolation_order
        source_row_count = round(tile_size / zoom_factor)
        source_row_margin = round(math.pow(2.0, math.ceil(math.log2(max(target_margin_size, 1.0 / zoom_factor)))))
        source_buffer_row_count = source_row_margin + source_row_count + source_row_margin
        target_row_margin = round(source_row_margin * zoom_factor)
        target_buffer_row_count = round(source_buffer_row_count * zoom_factor)

        source_row_array = np.zeros(shape=(input_image.channels, source_buffer_row_count, input_image.shapes[processing_level][1]), dtype=input_image.dtype)
        target_row_array = np.zeros(shape=(input_image.channels, target_buffer_row_count, zoom_output_width), dtype=input_image.dtype)
        target_zoom = (1.0, zoom_factor, zoom_factor)

        # Process the first tile row.
        #
        source_row_array[:, source_row_margin:, :] = input_image.read(spacing=input_image.spacings[processing_level],
                                                                      row=0,
                                                                      col=0,
                                                                      height=source_row_count + source_row_margin,
                                                                      width=input_image.shapes[processing_level][1])

        source_row_array[:, :source_row_margin, :] = np.flip(m=source_row_array[:, source_row_margin:source_row_margin + source_row_margin, :], axis=1)
        scipy.ndimage.zoom(input=source_row_array, zoom=target_zoom, output=target_row_array, order=interpolation_order, mode='reflect', prefilter=use_pre_filter)

        # Write out the first row.
        #
        for col in range(0, target_shape[1], tile_size):
            image_writer.write(tile=target_row_array[:, target_row_margin:target_row_margin + tile_size, col:col + tile_size], row=0, col=col)

        target_row_index = tile_size
        for row in range(source_row_count, input_image.shapes[processing_level][0], source_row_count):
            # Shuffle the rows upwards in the source row data and read a new row.
            #
            source_row_array[:, :source_row_margin, :] = source_row_array[:, source_row_count:source_row_count + source_row_margin, :]
            source_row_array[:, source_row_margin:, :] = input_image.read(spacing=input_image.spacings[processing_level],
                                                                          row=row,
                                                                          col=0,
                                                                          height=source_row_count + source_row_margin,
                                                                          width=input_image.shapes[processing_level][1])

            # Mirror the bottom of the read part if the image is smaller than the part to read.
            #
            empty_row_count = max(0, row + source_row_count + source_row_margin - input_image.shapes[processing_level][0])
            if 0 < empty_row_count:
                first_empty = source_buffer_row_count - empty_row_count
                mirror_length = min(empty_row_count, first_empty)

                source_row_array[:, first_empty:first_empty + mirror_length, :] = np.flip(m=source_row_array[:, first_empty - mirror_length:first_empty, :], axis=1)

            # Zoom the central tile row.
            #
            scipy.ndimage.zoom(input=source_row_array, zoom=target_zoom, output=target_row_array, order=interpolation_order, mode='reflect', prefilter=use_pre_filter)

            # Write out the row.
            #
            for col in range(0, target_shape[1], tile_size):
                image_writer.write(tile=target_row_array[:, target_row_margin:target_row_margin + tile_size, col:col + tile_size], row=target_row_index, col=col)

            # Increment the target row index.
            #
            target_row_index += tile_size

        # Finalize the output image and close the input image.
        #
        image_writer.close()

        if not isinstance(image, dptimagereader.ImageReader):
            input_image.close()

        # Log execution time.
        #
        execution_time = time.time() - start_time
        logger.debug('Done in {delta}'.format(delta=datetime.timedelta(seconds=execution_time)))

    else:
        logger.info('Skipping, target file already exits: {path}'.format(path=output_path))

#----------------------------------------------------------------------------------------------------

def zoom_image_to_template(image,
                           template,
                           output_path,
                           image_level,
                           image_pixel_spacing,
                           template_level,
                           template_pixel_spacing,
                           spacing_tolerance,
                           interpolation_order=0,
                           jpeg_quality=None,
                           overwrite=True):
    """
    Zoom the given input image from the given level to the shape of the template image at the given level.

    Args:
        image (dptimagereader.ImageReader, str): Input image object or path. The image that will be zoomed.
        template (dptimagereader.ImageReader, str): Template image object or path. The image will be used to extract target shape information.
        output_path (str): Output image path. The zoomed image.
        image_level (int, None): Level of the input image to use. Either this or the input_pixel_spacing parameter is required.
        image_pixel_spacing (float, None): Pixel spacing of the input image to use. (micrometer). Either this or the input_level parameter is required.
        template_level (int, None): Level of the template image to use. Either this or the template_pixel_spacing parameter is required.
        template_pixel_spacing (float, None): Pixel spacing of the template image to use. (micrometer). Either this or the template_level parameter is required.
        spacing_tolerance (float): Pixel spacing tolerance (percentage).
        interpolation_order (int): Interpolation order for zooming from the [0, 5] interval.
        jpeg_quality (int, None): JPEG quality (1-100) when using JPEG as compression method. If not set, the default of ImageWriter is used.
        overwrite (bool): If true existing targets will be overwritten.

    Raises:
        InvalidPixelSpacingValueError: Both image level and pixel spacing are None.
        InvalidZoomFactorError: Non-positive zoom factor.
        InvalidZoomTargetShapeError: The target shape to zoom to is invalid.
        InvalidInterpolationOrderError: Interpolation order out of [0, 5] bounds.
        AsymmetricZoomFactorError: Asymmetric zoom values.

        DigitalPathologyConfigError: Configuration errors.
        DigitalPathologyImageError: Image errors.
    """

    # Log paths.
    #
    logger = logging.getLogger(name=__name__)

    logger.info('Template image: {path}'.format(path=template.path if isinstance(template, dptimagereader.ImageReader) else template))

    # Get the template shape.
    #
    template_image = template if isinstance(template, dptimagereader.ImageReader) else dptimagereader.ImageReader(image_path=template,
                                                                                                                  spacing_tolerance=spacing_tolerance,
                                                                                                                  input_channels=None,
                                                                                                                  cache_path=None)

    template_level_to_use = template_level if template_level is not None else template_image.level(spacing=template_pixel_spacing)
    template_shape = template_image.shapes[template_level_to_use]

    # Zoom the image to the target shape.
    #
    zoom_image(image=image,
               output_path=output_path,
               zoom=template_shape,
               level=image_level,
               pixel_spacing=image_pixel_spacing,
               spacing_tolerance=spacing_tolerance,
               round_shape=True,
               interpolation_order=interpolation_order,
               jpeg_quality=jpeg_quality,
               overwrite=overwrite)

    # Close the opened template image.
    #
    if not isinstance(template, dptimagereader.ImageReader):
        template_image.close()

#----------------------------------------------------------------------------------------------------

def zoom_image_to_template_batch(job_list,
                                 image_level,
                                 image_pixel_spacing,
                                 template_level,
                                 template_pixel_spacing,
                                 spacing_tolerance,
                                 interpolation_order=0,
                                 jpeg_quality=None,
                                 overwrite=True):
    """
    Zoom the given input image to the resolution of the given level of the template image in batch mode. The pixel size of the result image will be the
    pixel size of the template image. The zoom factor is calculated from the matrix size (pixel count) difference and not the pixel size.

    Args:
        job_list (list): List of job (input path, output path) tuples.
        image_level (int, None): Level of the input image to use. Either this or the input_pixel_spacing parameter is required.
        image_pixel_spacing (float, None): Pixel spacing of the input image to use. (micrometer). Either this or the input_level parameter is required.
        template_level (int, None): Level of the template image to use. Either this or the template_pixel_spacing parameter is required.
        template_pixel_spacing (float, None): Pixel spacing of the template image to use. (micrometer). Either this or the template_level parameter is required.
        spacing_tolerance (float, None): Pixel spacing tolerance (percentage).
        interpolation_order (int): Interpolation order for zooming from the [0, 5] interval.
        jpeg_quality (int, None): JPEG quality (1-100) when using JPEG as compression method. If not set, the default of ImageWriter is used.
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

    logger.info('Zooming images to templates in batch mode...')
    logger.info('Job count: {count}'.format(count=len(job_list)))

    # Go through the job list.
    #
    failed_collection = []
    successful_collection = []
    for job_index in range(len(job_list)):
        # Get one job item.
        #
        input_path, template_path, output_path = job_list[job_index]

        try:
            # Print data paths if the mode is not a single file mode.
            #
            logger.info('Processing [{index}/{count}]: {path}'.format(index=job_index+1, count=len(job_list), path=output_path))

            # Zoom the image.
            #
            zoom_image_to_template(image=input_path,
                                   template=template_path,
                                   output_path=output_path,
                                   image_level=image_level,
                                   image_pixel_spacing=image_pixel_spacing,
                                   template_level=template_level,
                                   template_pixel_spacing=template_pixel_spacing,
                                   spacing_tolerance=spacing_tolerance,
                                   interpolation_order=interpolation_order,
                                   jpeg_quality=jpeg_quality,
                                   overwrite=overwrite)

        except Exception as exception:
            # Add case to the error collection.
            #
            failed_collection.append(output_path)
            logger.error('Error: {exception}'.format(exception=exception))

        else:
            # Save the result to the list of successful zooms.
            #
            successful_collection.append(output_path)

    # Log execution time.
    #
    execution_time = time.time() - start_time
    logger.debug('Zooming to templates batch done in {delta}'.format(delta=datetime.timedelta(seconds=execution_time)))

    # Return a list of successful zooms.
    #
    return successful_collection, failed_collection

#----------------------------------------------------------------------------------------------------

def save_image_at_level(image, output_path, level, pixel_spacing, spacing_tolerance, jpeg_quality=None, overwrite=True):
    """
    Save image at the required level.

    Args:
        image (dptimagereader.ImageReader, str): Input image object or path. The image that will be zoomed.
        output_path (str): Output image path.
        level (int, None): Processing level. Either this or the pixel_spacing parameter is required.
        pixel_spacing (float, None): Processing pixel spacing (micrometer). Either this or the level parameter is required.
        spacing_tolerance (float, None): Pixel spacing tolerance (percentage).
        jpeg_quality (int, None): JPEG quality (1-100) when using JPEG as compression method. If not set, the default of ImageWriter is used.
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

    # Log parameters.
    #
    logger = logging.getLogger(name=__name__)

    logger.info('Saving image at level...')
    logger.info('Input image: {path}'.format(path=image.path if isinstance(image, dptimagereader.ImageReader) else image))
    logger.info('Result image: {path}'.format(path=output_path))

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

        # Calculate source level and target spacing and add missing spacing information.
        #
        processing_level = input_image.level(spacing=pixel_spacing) if level is None else level

        if level is None:
            logger.debug('Identified level: {level}'.format(level=processing_level))

        if any(level_spacing is None for level_spacing in input_image.spacings):
            input_image.correct(spacing=1.0, level=0)

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
                                                  jpeg_quality=jpeg_quality,
                                                  empty_value=0,
                                                  skip_empty=None,
                                                  cache_path=None)

        # Load patch and write it out.
        #
        image_shape = input_image.shapes[processing_level]

        for row in range(0, image_shape[0], tile_size):
            for col in range(0, image_shape[1], tile_size):
                image_writer.write(tile=input_image.read(spacing=input_image.spacings[processing_level], row=row, col=col, height=tile_size, width=tile_size), row=row, col=col)

        # Finalize the output image.
        #
        image_writer.close()

        if not isinstance(image, dptimagereader.ImageReader):
            input_image.close()

        # Log execution time.
        #
        execution_time = time.time() - start_time
        logger.debug('Done in {delta}'.format(delta=datetime.timedelta(seconds=execution_time)))

    else:
        logger.info('Skipping, target file already exits: {path}'.format(path=output_path))

#----------------------------------------------------------------------------------------------------

def save_image_at_level_batch(job_list, level, pixel_spacing, spacing_tolerance, jpeg_quality=None, overwrite=True):
    """
    Save image at the required level in batch mode.

    Args:
        job_list (list): List of job (input path, output path) tuples.
        level (int, None): Processing level. Either this or the pixel_spacing parameter is required.
        pixel_spacing (float, None): Processing pixel spacing (micrometer). Either this or the level parameter is required.
        spacing_tolerance (float, None): Pixel spacing tolerance (percentage).
        jpeg_quality (int, None): JPEG quality (1-100) when using JPEG as compression method. If not set, the default of ImageWriter is used.
        overwrite (bool): If true existing targets will be overwritten.
        logger (logging.Logger, None): Logger object.

    Returns:
        (list, list): List of successfully processed items and list of failed items.
    """

    # Start time measurement.
    #
    start_time = time.time()

    # Log batch processing.
    #
    logger = logging.getLogger(name=__name__)

    logger.info('Saving images at level in batch mode...')
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
            logger.info('Processing [{index}/{count}]: {path}'.format(index=job_index+1, count=len(job_list), path=output_path))

            # Save image at the given level.
            #
            save_image_at_level(image=input_path,
                                output_path=output_path,
                                level=level,
                                pixel_spacing=pixel_spacing,
                                spacing_tolerance=spacing_tolerance,
                                jpeg_quality=jpeg_quality,
                                overwrite=overwrite)

        except Exception as exception:
            # Add case to the error collection.
            #
            failed_collection.append(output_path)
            logger.error('Error: {exception}'.format(exception=exception))

        else:
            # Save the result to the list of successful exports.
            #
            successful_collection.append(output_path)

    # Log execution time.
    #
    execution_time = time.time() - start_time
    logger.debug('Image saving at level batch done in {delta}'.format(delta=datetime.timedelta(seconds=execution_time)))

    # Return a list of successful exports.
    #
    return successful_collection, failed_collection
