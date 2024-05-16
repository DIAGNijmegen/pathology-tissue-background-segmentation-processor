"""
This module can dilate a given binary image and save it.
"""

from ..errors import processingerrors as dptprocessingerrors
from ..image import imagereader as dptimagereader
from ..image import imagewriter as dptimagewriter

import scipy.ndimage
import numpy as np
import logging
import os
import datetime
import time

#----------------------------------------------------------------------------------------------------

def dilate_image(image, output_path, dilation_distance, overwrite=True):
    """
    Dilate image.

    Args:
        image (dptimagereader.ImageReader, str): Input image object or path.
        output_path (str): Output image path. The zoomed image.
        dilation_distance (float): Dilation distance (micrometers).
        overwrite (bool): If true existing targets will be overwritten.

    Raises:
        InvalidDilationDistanceError: Invalid dilation iteration count.
        NonMonochromeInputImageError: The input image is not monochrome.

        DigitalPathologyConfigError: Configuration errors.
        DigitalPathologyImageError: Image errors.
    """

    # Start time measurement.
    #
    start_time = time.time()

    # Log parameters.
    #
    logger = logging.getLogger(name=__name__)

    logger.info('Dilating image...')
    logger.info('Input image: {path}'.format(path=image.path if isinstance(image, dptimagereader.ImageReader) else image))
    logger.info('Result image: {path}'.format(path=output_path))
    logger.info('Dilation distance: {distance} um'.format(distance=dilation_distance))

    # Check if the dilation distance is valid.
    #
    if dilation_distance <= 0.0:
        raise dptprocessingerrors.InvalidDilationDistanceError(dilation_distance)

    # Check if the target image already exits.
    #
    if not os.path.isfile(output_path) or overwrite:
        # Open image.
        #
        input_image = image if isinstance(image, dptimagereader.ImageReader) else dptimagereader.ImageReader(image_path=image, spacing_tolerance=0.25, input_channels=None, cache_path=None)

        # Check if image is monochrome. Cannot threshold color images.
        #
        if input_image.coding != 'monochrome':
            raise dptprocessingerrors.NonMonochromeInputImageError(input_image.path, input_image.coding)

        if any(level_spacing is None for level_spacing in input_image.spacings):
            input_image.correct(spacing=1.0, level=0)

        # Calculate dilation iterations.
        #
        dilation_iters = round(dilation_distance / input_image.spacings[0])
        logger.info('Dilation iterations: {iterations}'.format(iterations=dilation_iters))

        # Configure the image writer.
        #
        tile_size = 512
        image_writer = dptimagewriter.ImageWriter(image_path=output_path,
                                                  shape=input_image.shapes[0],
                                                  spacing=input_image.spacings[0],
                                                  dtype=np.uint8,
                                                  coding=input_image.coding,
                                                  compression=None,
                                                  interpolation=None,
                                                  tile_size=tile_size,
                                                  jpeg_quality=None,
                                                  empty_value=0,
                                                  skip_empty=None,
                                                  cache_path=None)

        # Check if the dilation iteration count is larger than zero:
        #
        if dilation_iters:
            # Initialize margin content.
            #
            bottom_margin = np.zeros(shape=(input_image.channels, dilation_iters, input_image.shapes[0][1]), dtype=input_image.dtype)

            # Process the rows.
            #
            for row in range(0, input_image.shapes[0][0], tile_size):
                # Read one row plus the margin in the bottom.
                #
                tile_row_array = input_image.read(spacing=input_image.spacings[0], row=row, col=0, height=tile_size + dilation_iters, width=input_image.shapes[0][1])

                # Dilate the content and combine the top of the row with the bottom margin of the previous row.
                #
                tile_row_array = scipy.ndimage.binary_dilation(input=tile_row_array, iterations=dilation_iters)
                tile_row_array[:, :dilation_iters, :] = np.logical_and(tile_row_array[:, :dilation_iters, :], bottom_margin)

                # Write out the first row.
                #
                for col in range(0, input_image.shapes[0][1], tile_size):
                    image_writer.write(tile=tile_row_array[:, :tile_size, col:col + tile_size].astype(dtype=np.uint8), row=row, col=col)

                # Save the bottom margin for the next row.
                #
                bottom_margin = np.copy(tile_row_array[:, -dilation_iters:, :])

        else:
            # The iteration count is 0, the dilated image is identical to the original.
            #
            for row in range(0, input_image.shapes[0][0], tile_size):
                for col in range(0, input_image.shapes[0][1], tile_size):
                    image_writer.write(tile=input_image.read(spacing=input_image.spacings[0], row=row, col=col, height=tile_size, width=tile_size), row=row, col=col)

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

def dilate_image_batch(job_list, dilation_distance, overwrite=True):
    """
    Dilate image in batch mode.

    Args:
        job_list (list): List of job (input path, output path) tuples.
        dilation_distance (float): Dilation distance (micrometers).
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

    logger.info('Dilating images in batch mode...')
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

            # Apply the threshold.
            #
            dilate_image(image=input_path, output_path=output_path, dilation_distance=dilation_distance, overwrite=overwrite)

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
    logger.debug('Image dilation batch done in {delta}'.format(delta=datetime.timedelta(seconds=execution_time)))

    # Return a list of successful thresholds.
    #
    return successful_collection, failed_collection
