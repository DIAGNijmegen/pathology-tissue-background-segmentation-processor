"""
This file contains a class for applying models on whole-slide images.
"""

from . import imageclassifier as dptimageclassifier
from ..image import imagewriter as dptimagewriter
import numpy as np

#----------------------------------------------------------------------------------------------------

class WholeSlideImageClassifier(dptimageclassifier.ImageClassifier):
    """
    Classifies whole slide images from disk.

    This class applies a fully convolutional model tiled fashion to
    a whole-slide image on disk.

    Note: the mask image is always assumed to have the same size as the
    original image at the level corresponding to output scale.
    """
    def __init__(self):
        super().__init__()

        self._result_file_path = ""

        self._writer = None
        self._level = 0
        self._mask_level = 0
        self._write_tile_size = 512

        # The class keeps results in a buffer until it can write a row of write_tile_size
        # For this to work we need to keep track which row we are currently doing inference on
        #
        self._buffer_row = None
        self._buffer_written = False
        self._buffer_row_index = 0
        self._current_row_index = 0
        self._last_tile_shape = None
        self._tiff_row = -1

    def _check_parameters(self):
        """
        This function checks whether all the parameters are provided correctly.

        Small extension to base class functionality as WholeSlideImageClassifier only
        supports tiled analysis.
        """
        checked = super(WholeSlideImageClassifier, self)._check_parameters()
        if self._tile_size < 1:
            self._logger.error("WholeSlideImageClassifier only supports tiled analysis")
            return False
        return checked

    def set_input(self, image, level=0, channels=None, mask=None, mask_level=0):
        """
        Provide the input image(s) for the ImageClassifier to process

        In the multi-resolution case, images can be provided as having the same
        size with the same center pixel or cropped to the appropriate size with
        the same center pixel. In the former case, the classifier will
        determine the correct areas to use.
        A mask image is optional, this masks out all zero values of the mask in
        the result image. Mask is expected to be of the same size as the image
        at the output_scale.
        """
        self._input_channels = channels
        if mask:
            self._mask = mask
        else:
            self._mask = None
        self._mask_level = mask_level
        self._level = int(level)
        self._input = image

    def set_result_file_path(self, path):
        """
        Set the path to the result file on disk.
        """
        self._result_file_path = path

    def get_result_file_path(self):
        """
        Returns the path to the result file on disk.
        """
        return self._result_file_path

    def _get_input_image_shape(self):
        """
        Get input image shapes for the defined level.

        In a WSIImage, the different scales are within the same image file at
        different levels. As such we acquire the dimensions at the different
        levels.
        """
        return tuple(reversed(self._input.shapes[self._level]))

    def _get_mask_shape(self):
        """
        Get the mask dimensions.
        """
        return tuple(reversed(self._mask.shapes[self._mask_level]))

    def _get_image_data_from_input(self, box=[], scale=1):
        """
        Get pixel data from the image region defined by box at specified scale.

        Corrects the box specified for the corresponding image level the data
        has to be acquired from. Then obtain the data from the image file and
        return it.
        """

        image_box = self._input.read(spacing=self._input.spacings[self._level],
                                     row=box[0],
                                     col=box[2],
                                     height=int(box[1] - box[0]),
                                     width=int(box[3] - box[2]))
        image_box = image_box.transpose((1, 2, 0))

        if self._input_channels:
            image_box = image_box[:, :, self._input_channels]

        return image_box

    def _get_image_data_from_mask(self, box=[]):
        """
        Get mask data from the image file from the area defined by box.
        """

        downscale = self._input.shapes[self._level][1] / self._mask.shapes[self._mask_level][1]
        mask_box = self._mask.read(spacing=self._mask.spacings[self._mask_level],
                                   row=int(box[0] // downscale),
                                   col=int(box[2] // downscale),
                                   height=int((box[1] - box[0]) // downscale),
                                   width=int((box[3] - box[2]) // downscale))
        mask_box = mask_box.transpose((1, 2, 0))

        return mask_box

    def _initialize_result(self):
        """
        Initializes the result image on disk based on provided parameters.

        Instantiates the image file on disk and sets attributes like data type,
        number of channels and color type.
        """

        # The class uses a buffer as explained in __init__, it's initialized as twice output tile size
        # or write size, whichever is bigger
        #
        if self._tile_size > self._write_tile_size:
            buffer_height = self._tile_size * 2
        else:
            buffer_height = self._write_tile_size + self._tile_size

        shp = self._get_input_image_shape()

        if not self._soft or self._output_class >= 0:
            if self._soft:
                if self._channels_in_result < 0:
                    self._buffer_row = np.zeros((int(buffer_height), shp[0], self.model.getnumberofoutputchannels()))
                else:
                    self._buffer_row = np.zeros((int(buffer_height), shp[0], self._channels_in_result))
            else:
                self._buffer_row = np.zeros((int(buffer_height), shp[0]))
            self._channels_in_result = 1
        elif self._channels_in_result < 0 and self._output_class < 0:
            self._channels_in_result = self.model.getnumberofoutputchannels()
            self._buffer_row = np.zeros((int(buffer_height), shp[0], self._channels_in_result))

        if self._soft:
            data_type = np.uint8 if self._quantize else np.float32
            color_coding = 'indexed'
            indexed_channels = self._channels_in_result
            interpolation_mode = 'linear'
        else:
            data_type = np.uint8
            color_coding = 'monochrome'
            indexed_channels = 0
            interpolation_mode = 'nearest'

        self._writer = dptimagewriter.ImageWriter(image_path=str(self._result_file_path),
                                                  shape=(int(shp[1]), int(shp[0])),
                                                  spacing=self._input.spacings[self._level],
                                                  dtype=data_type,
                                                  coding=color_coding,
                                                  indexed_channels=indexed_channels,
                                                  compression='lzw',
                                                  interpolation=interpolation_mode,
                                                  tile_size=self._write_tile_size,
                                                  jpeg_quality=None,
                                                  empty_value=0,
                                                  skip_empty=True,
                                                  cache_path=None)

        return

    def _get_result_shape(self):
        """
        Returns the shape of the result image.
        """
        shp = self._get_input_image_shape()
        return [shp[1], shp[0], self._channels_in_result]

    def _write_tile_to_result(self, tile, row, col, shape):
        """
        This function gets called by the base class. It will fill the row buffer and if we have
        enough data send a row to be written to _write_row_to_result.
        """
        if row > self._current_row_index:
            self._current_row_index = row
            self._buffer_row_index += self._last_tile_shape[0]

        if self._write_tile_size < self._buffer_row_index:
            while self._write_tile_size < self._buffer_row_index:
                # Only write tiles if there is actually information there (indicated by _buffer_written)
                #
                if self._buffer_written:
                    self._write_row_to_result(self._buffer_row[0:self._write_tile_size])

                    # Move data in buffer up, (removing the written row)
                    #
                    new_buffer = np.zeros((self._buffer_row.shape))
                    new_buffer[0:self._buffer_row.shape[0] - self._write_tile_size] = self._buffer_row[self._write_tile_size:]
                    self._buffer_row = new_buffer

                    # If we have a mask and the buffer is empty we do not have to write tiles
                    #
                    if self._mask is not None:
                        if not np.any(self._buffer_row):
                            self._buffer_written = False
                else:
                    self._tiff_row += 1

                self._buffer_row_index -= self._write_tile_size

        # Calculate the coordinates of the tile with respect to the buffer
        #
        c_buffer_top = self._buffer_row_index
        c_buffer_bottom = c_buffer_top + shape[0]
        c_buffer_left = col
        c_buffer_right = c_buffer_left + shape[1]

        # We save the shape, because if the next result will be on a new row we will need
        # to shift in the buffer by our current tile height. We can not assume all tiles are the same shape
        # (tiles at borders are bigger)
        #
        self._last_tile_shape = shape

        if tile is not None:
            self._buffer_written = True
            self._buffer_row[c_buffer_top:c_buffer_bottom, c_buffer_left:c_buffer_right] = tile

    def _write_row_to_result(self, row):
        """
        Write a single tile to the result image.
        """
        if self._writer is not None:
            for y in range(0, row.shape[0], self._write_tile_size):
                self._tiff_row += 1
                for x in range(0, row.shape[1], self._write_tile_size):
                    tile = row[y: y + self._write_tile_size, x: x + self._write_tile_size]
                    self._write_tile_to_tiff(tile, x, self._tiff_row * self._write_tile_size)

    def _write_tile_to_tiff(self, tile, x, y):
        """
        Write a tile to tiff in the correct data format and on the designated location
        """

        if tile.shape[1] < self._write_tile_size or tile.shape[0] < self._write_tile_size:
            if self._soft:
                tile = np.pad(tile,
                              ((0, max(0, self._write_tile_size - tile.shape[0])),
                               (0, max(0, self._write_tile_size - tile.shape[1])), (0, 0)),
                              'constant')
            else:
                tile = np.pad(tile,
                              ((0, max(0, self._write_tile_size - tile.shape[0])),
                               (0, max(0, self._write_tile_size - tile.shape[1]))),
                              'constant')
        if self._soft:
            if self._output_class < 0:
                write_tile = tile[0:self._write_tile_size, 0:self._write_tile_size, :]
            else:
                write_tile = tile[0:self._write_tile_size, 0:self._write_tile_size, self._output_class:self._output_class + 1]

            if self._quantize:
                write_tile = np.clip(((write_tile - self._quantize_min) / (self._quantize_max - self._quantize_min)) * 255, 0, 255).astype(np.uint8)
            else:
                write_tile = write_tile.astype(np.float32)
        else:
            write_tile = tile[0:self._write_tile_size, 0:self._write_tile_size, None].astype(np.uint8)

        self._writer.write(tile=write_tile.transpose((2, 0, 1)), row=y, col=x)

    def _finish(self):
        """
        Write the result image pyramid and close the file.
        """
        if self._writer:
            # Empty the last row of the buffer and reset the buffer
            #
            self._write_row_to_result(self._buffer_row)
            self._buffer_row = np.empty((self._buffer_row.shape))
            self._buffer_row_index = 0
            self._current_row_index = 0

            self._writer.close()
