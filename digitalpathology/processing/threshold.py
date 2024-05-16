"""
This module can threshold a given image and save it as a binary.
"""

from ..errors import processingerrors as dptprocessingerrors
from ..image import imagereader as dptimagereader
from ..image import imagewriter as dptimagewriter

import numpy as np
import logging
import datetime
import time
import os

#----------------------------------------------------------------------------------------------------

def low_threshold_image(image, output_path, low_threshold, overwrite=True):
    """
    Low threshold image.

    Args:
        image (dptimagereader.ImageReader, str): Input image object or path.
        output_path (str): Output image path. The thresholded image.
        low_threshold (int, float): Low threshold value.
        overwrite (bool): If true existing targets will be overwritten.

    Raises:
        NonMonochromeInputImageError: The input image is not monochrome.
        ThresholdDimensionImageChannelCountMismatchError: Dimensions of the threshold does not match image channel count.

        DigitalPathologyConfigError: Configuration errors.
        DigitalPathologyImageError: Image errors.
    """

    # Start time measurement.
    #
    start_time = time.time()

    # Log parameters.
    #
    logger = logging.getLogger(name=__name__)

    logger.info('Thresholding image...')
    logger.info('Input image: {path}'.format(path=image.path if isinstance(image, dptimagereader.ImageReader) else image))
    logger.info('Result image: {path}'.format(path=output_path))
    logger.info('Low threshold: {threshold}'.format(threshold=low_threshold))

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

        # Apply threshold on image patch-by-patch.
        #
        for row in range(0, input_image.shapes[0][0], tile_size):
            for col in range(0, input_image.shapes[0][1], tile_size):

                # Load and threshold patch.
                #
                input_tile = input_image.read(spacing=input_image.spacings[0], row=row, col=col, height=tile_size, width=tile_size)
                result_tile = np.greater_equal(input_tile, low_threshold).astype(np.uint8)
                image_writer.write(tile=result_tile, row=row, col=col)

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

def low_threshold_image_batch(job_list, low_threshold, overwrite=True):
    """
    Low threshold image in batch mode.

    Args:
        job_list (list): List of job (input path, output path) tuples.
        low_threshold (float, list): Low threshold values. When the list if longer than 1, the values are used per channel.
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

    logger.info('Thresholding images in batch mode...')
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
            low_threshold_image(image=input_path, output_path=output_path, low_threshold=low_threshold, overwrite=overwrite)

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
    logger.debug('Image thresholding batch done in {delta}'.format(delta=datetime.timedelta(seconds=execution_time)))

    # Return a list of successful thresholds.
    #
    return successful_collection, failed_collection
