"""
This file contains a class for applying modelbase models on images.
"""

import diagmodels.models.modelbase as dmsmodelbase

import numpy as np
import skimage.transform
import logging
import sys

#----------------------------------------------------------------------------------------------------

class ImageClassifier(object):
    """
    Classify larger image using patch-trained convolution neural networks.

    The class provides two strategies to classify the images, either through
    a tiled classification approach or by using a fully convolutional network.
    If tile_size is 0 then it will feed the whole image to the network, otherwise
    it will use the tile-by-tile approach.

    The rest of the functionality and implementation details are covered in the
    docstrings of the specific member functions.
    """
    def __init__(self):

        super().__init__()

        self._logger = None
        self._model = None
        self._mask = None
        self._tile_size = 0
        self._axis_description = "whc"
        self._postprocess_function = None
        self._preprocess_function = None
        self._soft = True
        self._input = None
        self._input_channels = None
        self._channels_in_result = -1
        self._result = None
        self._output_class = -1
        self._quantize = False
        self._quantize_min = 0.
        self._quantize_max = 1.
        self._padding_mode = 'constant'
        self._padding_constant = 255
        self._verbose = True

        self.__init_logging()

    def __init_logging(self):
        """Initialize logging."""

        # Configure logging. This class relies on configured logging somewhere down on the hierarchy.
        #
        qualified_class_name = '{module_name}.{class_name}'.format(module_name=self.__class__.__module__, class_name=self.__class__.__name__)
        self._logger = logging.getLogger(name=qualified_class_name)

    ###########################
    # Setter/getter functions #
    ###########################
    def set_input(self, image, input_channels=None, mask=None):
        """
        Provide the input image for the ImageClassifier to process

        A mask image is optional, this masks out all zero values of the mask in
        the result image. Mask is expected to be of the same size as the image
        at the output_scale.
        """
        self._input = image
        self._input_channels = input_channels
        self._mask = mask

    @property
    def output_class(self):
        """
        Get the output class used for the result.
        """
        return self._output_class

    @output_class.setter
    def output_class(self, output_class):
        """
        Only keep the results of the specified class.
        """
        self._output_class = output_class

    @property
    def axis_description(self):
        """
        Returns the order of the axes as a string.
        """
        return self._axis_description

    @axis_description.setter
    def axis_description(self, axis_description):
        """
        Set the order of axes used throughout this class (source image/network output/results)
        """
        allowed_descriptions = ["whc", "cwh"]
        if axis_description in allowed_descriptions:
            self._axis_description = axis_description
        else:
            self._logger.error("Unsupported axis description, only supported options are: " + str(allowed_descriptions))

    @property
    def result(self):
        """
        Get the result image
        """
        return self._result

    @property
    def model(self):
        """
        Getter for the model
        """
        return self._model

    @model.setter
    def model(self, model):
        """
        Setter for the model
        """
        if isinstance(model, dmsmodelbase.ModelBase):
            self._model = model
        else:
            self._logger.error("Invalid model given, should be a Modelbase model")

    @property
    def tile_size(self):
        """
        Getter for the tile_size
        """
        return self._tile_size

    @tile_size.setter
    def tile_size(self, tile_size):
        """
        If analysis is to be performed per tile, provide a tile size
        """
        if tile_size > -1 and tile_size % 1 == 0:
            self._tile_size = tile_size
        else:
            self._logger.error("Invalid value for tile size, can only be 0 or a positive integer")

    @property
    def preprocess_function(self):
        """
        Gets the current function used to preprocess tiles to be classified.
        Note: this function is only used in the tiles approach.
        """
        return self._preprocess_function

    @preprocess_function.setter
    def preprocess_function(self, preprocess_function):
        """
        Sets a function to preprocess tiles to be classified.
        """
        self._preprocess_function = preprocess_function

    @property
    def postprocess_function(self):
        """
        Gets the current function used to postprocess classified tiles.
        Note: this function is only used in the tiles approach.
        """
        return self._postprocess_function

    @postprocess_function.setter
    def postprocess_function(self, postprocess_function):
        """
        Sets a function to postprocess classified tiles.
        """
        self._postprocess_function = postprocess_function

    @property
    def channels_in_result(self):
        """
        Gets the current function used to postprocess classified tiles.
        """
        return self._channels_in_result

    @channels_in_result.setter
    def channels_in_result(self, channels_in_result):
        """
        Sets a function to postprocess classified tiles.
        """
        self._channels_in_result = channels_in_result

    @property
    def soft(self):
        """
        Getter for soft
        """
        return self._soft

    @soft.setter
    def soft(self, soft):
        """
        Sets whether soft classification should be performed

        Soft classification results in a probability per class, hard
        classification only an overall label (argmax).
        """
        if soft is True or soft is False:
            self._soft = soft
        else:
            self._logger.error("Invalid value for soft, can only be True or False")

    @property
    def verbose(self):
        """
        Getter for verbose
        """
        return self._verbose

    @verbose.setter
    def verbose(self, verbose):
        """
        Set verbosity. If False all prints to screen are suppressed.
        """
        self._verbose = verbose

    def _get_input_image_shape(self):
        """
        Gets the input image shapes
        """
        return self._input.shape

    def _get_mask_shape(self):
        """
        Gets the shape of the mask
        """
        return self._mask.shape if self._mask is not None else tuple()

    @property
    def quantize(self):
        """
        Gets boolean if results gets clipped
        """
        return self._quantize

    @quantize.setter
    def quantize(self, quantize):
        """
        Set quantize (clipping final result)
        """
        self._quantize = quantize

    @property
    def quantize_min_and_max(self):
        """
        Get min and max of clipping in the case of quantize
        """
        return self._quantize_min, self._quantize_max

    @quantize_min_and_max.setter
    def quantize_min_and_max(self, qminmax):
        """
        Set min and max of clipping in the case of quantize
        """
        self._quantize_min = qminmax[0]
        self._quantize_max = qminmax[1]

    @property
    def padding_mode(self):
        """
        Get the padding mode.
        """
        return self._padding_mode

    @padding_mode.setter
    def padding_mode(self, mode):
        """
        Set the padding mode.
        """
        allowed_padding_modes = ["constant", "reflect"]
        if mode in allowed_padding_modes:
            self._padding_mode = mode
        else:
            self._logger.error("Unsupported padding mode, only supported options are: " + str(allowed_padding_modes))

    @property
    def padding_constant(self):
        """
        Get the padding constant.
        """
        return self._padding_constant

    @padding_constant.setter
    def padding_constant(self, value):
        """
        Set the padding constant.
        """
        self._padding_constant = value

    ########################################################
    # Functions which setup the actual processing pipeline #
    ########################################################
    def _check_parameters(self):
        """
        This function checks whether all the parameters are provided correctly
        """
        if self._input is None:
            self._logger.error("No input images given, cannot calculate results")
            return False
        if self._model is None:
            self._logger.error("No model given, cannot calculate results")
            return False
        return True

    #####################################################
    # Functions to get data from the input images/masks #
    #####################################################
    def _get_image_data_from_input(self, box=None):
        """
        Get pixel data from the input arrays
        """
        if box:
            tile = self._input[box[0]:box[1], box[2]:box[3], self._input_channels] if self._input_channels \
                else self._input[box[0]:box[1], box[2]:box[3], :]
        else:
            tile = self._input[:, :, self._input_channels] if self._input_channels else self._input
        return tile

    def _get_image_data_from_mask(self, box=None):
        """
        Get pixel data from the mask array, corrected for the input crops.
        """
        if box:
            return self._mask[box[0]:box[1], box[2]:box[3]]
        else:
            return self._mask

    #######################################
    # Functions handling the result image #
    #######################################
    def _initialize_result(self):
        """
        Initialize the right result shape
        """
        nr_channels = 1
        out_dtype = "ubyte"
        if self._soft and self._output_class < 0:
            if self._channels_in_result == -1:
                self._channels_in_result = self.model.getnumberofoutputchannels()
            nr_channels = self._channels_in_result
            out_dtype = "float32"
        elif self._soft and not self._quantize:
            out_dtype = "float32"

        if self._axis_description == "cwh":
            self._result = np.squeeze(np.zeros((nr_channels,
                                                self._get_input_image_shape()[0],
                                                self._get_input_image_shape()[1]),
                                               dtype=out_dtype))
        else:
            self._result = np.squeeze(np.zeros((self._get_input_image_shape()[0],
                                                self._get_input_image_shape()[1],
                                                nr_channels), dtype=out_dtype))
        return

    def _get_result_shape(self):
        """
        Get the shape of the results
        """
        if self._result is not None:
            return self._result.shape

    def _finish(self):
        pass

    #############################################
    # Functions for one-go-processing of images #
    #############################################
    def process(self):
        """
        Run the processing given inputs and parameters
        """
        self._result = None
        if self._check_parameters():
            if self._tile_size == 0:
                self._result = self._process_image()
            else:
                self._process_image_tiled()
            if self._output_class >= 0 and self._result:
                self._result = self._result[self._output_class]
            return True
        else:
            self._logger.warning("One or more parameter values were invalid")
            return False

    def _process_image(self, image=None, mask=None, rec_info=None):
        """
        Performs classification of an image in one go.

        This function using the prediction functions to classify the image.
        If image is not given to this function, it will fill this variable
        from self._inputs. Optionally, msk_data can be provided to mask
        out certain parts of the image.
        """

        # Get image and mask data from self._input if it is not provided by
        # the caller of this function.
        #
        if image is None:
            image = self._get_image_data_from_input()

        # Determine the amount of padding required for each side of the image.
        # Even filters lose more pixels at the right edge, depending on the
        # padding settings, as do pooling layers. This needs to be corrected.
        if not rec_info:
            pads, downsamples, interpolation_lost = self._model.getreconstructioninformation(input_shape=image.shape)
        else:
            pads, downsamples, interpolation_lost = rec_info

        # Tranpose image if channels are first
        #
        if self._axis_description == "cwh":
            image = image.transpose(2, 1, 0)

        result = self._model.predict(image[None])['predictions']
        if isinstance(result, list):
            result = result[0]

        if len(result.shape) == 2:
            result = result[None]
        else:
            result = result[0]

        # Transpose back to channels last -> rest of the code expects channels last.
        #
        if self._axis_description == "cwh":
            result = result.transpose(2, 1, 0)

        if not self._soft:
            result = (np.argmax(result, axis=-1) + 1)[:, :, None]

        # Zoom the result back up to the original image size minus the padding
        # We need to use edge mode here so the edge is not influence by zeros
        #
        zoomed_result = np.array([skimage.transform.rescale(result[:, :, c], (downsamples[0], downsamples[1]),
                                                            preserve_range=True, order=self._soft, mode="edge")
                                  for c in range(result.shape[2])], dtype="float32")

        zoomed_result = np.squeeze(zoomed_result)

        # Add the padding, the pixels lost due to the convolution and pooling
        # operations. This should allow the image to align nicely with the
        # input image.
        #
        if self._soft:
            padded_result = np.pad(zoomed_result, ((0, 0), (pads[2], pads[3]), (pads[0], pads[1])), 'constant')
            if mask is not None:
                mask = np.repeat(mask, padded_result.shape[2], 0)
                padded_result *= mask
        else:
            padded_result = np.pad(zoomed_result, ((pads[2], pads[3]), (pads[0], pads[1])),
                                   'constant')
            if mask is not None:
                padded_result = (padded_result + 1) * mask

        if self._soft:
            padded_result = padded_result.transpose((1, 2, 0))

        return padded_result

    def _process_image_tiled(self):
        """
        Process the image tile-by-tile feeding tiles of self._tile_size to the network.
        """
        network_lost, network_downsamples, interpolation_lost = \
            self._model.getreconstructioninformation(input_shape=(self._tile_size, self._tile_size, 0))

        if self._soft:
            interpolation_lost = np.ceil(interpolation_lost * 2).astype(int)
        else:
            interpolation_lost = np.ceil(interpolation_lost).astype(int)

        # Initialize output
        #
        self._initialize_result()

        # Determine output shape + width & height
        #
        result_shape = self._get_result_shape()

        result_width = result_shape[1]
        result_height = result_shape[0]

        # check if tile size minus everything that is lost is more then 0
        #
        if self._tile_size - network_lost[0] - network_lost[1] - interpolation_lost[0] - interpolation_lost[1] <= 0:
            raise ValueError("Tile_size " + str(self._tile_size) + " < than the amount of pixels lost on the edge (" +
                             str(network_lost[0] + network_lost[1] + interpolation_lost[0] + interpolation_lost[1]) + "px)."
                             " Try a bigger tile-size.")

        # Determine tile-, cropped-  height and width + the range
        tile_height = self._tile_size
        tile_width = self._tile_size

        # What is lost and what isn't:
        #
        #
        #  | --------------------------- image ----------------------------- |
        #
        #  | ------------ input ------------ |
        #  | lost_l+i -- result  -- lost_r+i |                                        > lost_l = lost due to network, lost_l+i = lost due to network + interpolation
        #  | =========|= cropped =|                                                   > note: we should save lost_l for the first tile
        #                                                                             > cropped_width = input_width - lost_l+i - lost_r+i
        #             | ------------- next  ----------- |                             > tile_x = lost_l_i + cropped_width * 1 - lost_l+i
        #             | lost_l+i -- result  -- lost_r+i |
        #                          |= cropped =|
        #
        #                          | ------------ next  ------------ |                > tile_x = lost_l_i + cropped_width * 2 - lost_l+i
        #                          | lost_l+i -- result  -- lost_r+i |
        #                                     |= cropped =|
        #
        #                                     | ------------ last  +++++++++ |        > + = pad_right
        #                                     | lost_l+i -- result  -- (input_width - pad_right) |
        #                                                 |= cropped =|======|        > extra crop right = (cropped_width * 4 + lost_r_i) - result_width
        #

        cropped_height = self._tile_size - network_lost[2] - network_lost[3] - interpolation_lost[2] - interpolation_lost[3]
        cropped_width = self._tile_size - network_lost[0] - network_lost[1] - interpolation_lost[0] - interpolation_lost[1]

        lost_l_i = network_lost[0] + interpolation_lost[0]  # amount of valid result lost at the left
        lost_r_i = network_lost[1] + interpolation_lost[1]  # amount of valid result lost at the right
        lost_t_i = network_lost[2] + interpolation_lost[2]  # etc.
        lost_b_i = network_lost[3] + interpolation_lost[3]

        y_range = range(-network_lost[0], result_height, cropped_height)  # add pad left
        x_range = range(-network_lost[2], result_width, cropped_width)  # add pad top

        # Calculate
        y_ticks = 50
        y_step = len(y_range) / y_ticks
        y_prints = set(y_range[index] for index in [round(index * y_step) - 1 for index in range(1, y_ticks + 1)])

        # Define padding keywords. Numpy is sensitive to superfluous keywords
        padding_kwargs = {'constant_values': self._padding_constant} if self._padding_mode == 'constant' else {}

        for y in y_range:
            if self._verbose:
                if y in y_prints:
                    sys.stdout.write('*')
                    sys.stdout.flush()

            for x in x_range:
                #
                # tile_{x|y} = coordinates of tile to fetch from input
                # tile_result_{x|y} = coordinates of the resulting network output tile
                # crop_{left|right|top|bottom} = what should be cropped from the network output
                # pad_{left|right|top|bottom} = what should be padded to network input tile
                #

                # Define cropbox
                #
                crop_left = lost_l_i
                crop_right = lost_r_i
                crop_top = lost_t_i
                crop_bottom = lost_b_i

                pad_left = 0
                pad_right = 0
                pad_top = 0
                pad_bottom = 0

                # 'Edge'-case when on the left border
                #
                if x < 0:
                    tile_x = 0
                    # Because in valid-padding networks we loose output on the left *and*
                    # we want the have the same output size as input size, 'x' starts at negative values
                    # corresponding to how much we loose on the left. Since the input image has no pixels
                    # at negative coordinates, we will pad zeros.
                    # It could be that after one tile we are still in the 'negative' coordinates because
                    # the output-width of our network is smaller than the output-lost at the border. So
                    # we need to pad multiple tiles. 
                    tile_result_x = network_lost[0] + x  # since x is negative this works 
                    pad_left = -x  # pad zeros for the amount of input image we do not have
                    crop_left = network_lost[0]  # We want to keep interpolation lost at the left border
                # 'Edge'-case when on the right border
                #
                elif x + self._tile_size >= result_width:
                    tile_x = x
                    tile_result_x = crop_left + tile_x
                    # The tile we want to give to the network is larger then the input
                    # This means we have to pad the tile to get to the right size
                    # (result_width here is the same as input_width)
                    #
                    pad_right = (tile_x + self.tile_size) - result_width
                    # If this is the last tile we probably need to crop more on the right
                    # to end up with exactly the width of the final result
                    #
                    # To check if this is the last, check if the valid network result > result_width
                    # (self._tile_size - crop_left - network_lost[1]) is basically:
                    #  = valid output + interpolation lost at the right (since we want to keep it there)
                    #
                    if tile_result_x + (self._tile_size - crop_left - network_lost[1]) >= result_width:
                        crop_right = tile_result_x + (self._tile_size - crop_left) - result_width
                # Normal case
                #
                else:
                    tile_x = x
                    # Coordinates in the result will be input tile coords + what has been cropped left
                    tile_result_x = crop_left + tile_x

                # For explanations see logic above
                #
                if y < 0:
                    tile_y = 0
                    tile_result_y = network_lost[2] + y
                    pad_top = -y
                    crop_top = network_lost[2]
                elif y + self._tile_size >= result_height:
                    tile_y = y
                    tile_result_y = crop_top + tile_y
                    pad_bottom = (tile_y + self.tile_size) - result_height
                    if tile_result_y + self._tile_size - crop_top - network_lost[3] >= result_height:
                        crop_bottom = tile_result_y + (self._tile_size - crop_top) - result_height
                else:
                    tile_y = y
                    tile_result_y = crop_top + tile_y

                tile_result_width = self._tile_size - crop_right - crop_left
                tile_result_height = self._tile_size - crop_top - crop_bottom

                # If there is a mask fetch the data
                #
                msk_tile = None
                if self._mask is not None:
                    msk_tile = self._get_image_data_from_mask([tile_y,
                                                               tile_y + tile_height,
                                                               tile_x,
                                                               tile_x + tile_width])

                    if not msk_tile.any():
                        if not self._soft:
                            self._write_tile_to_result(None,
                                                       tile_result_y,
                                                       tile_result_x,
                                                       shape=(tile_result_height, tile_result_width))
                        else:
                            self._write_tile_to_result(None,
                                                       tile_result_y,
                                                       tile_result_x,
                                                       shape=(tile_result_height,
                                                              tile_result_width,
                                                              result_shape[2]))
                        continue

                image_tile = self._get_image_data_from_input([tile_y, tile_y + tile_height - pad_bottom - pad_top, tile_x, tile_x + tile_width - pad_right - pad_left])
                image_tile = np.pad(image_tile, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode=self._padding_mode, **padding_kwargs)

                if self._mask is not None:
                        msk_tile = skimage.transform.resize(msk_tile,
                                                            (image_tile.shape[0], image_tile.shape[1]),
                                                            order=0, mode='constant')

                if self._input_channels:
                    image_tile = image_tile[:, :, self._input_channels]

                if self._preprocess_function:
                    image_tile = self._preprocess_function(image_tile)

                padded_result = self._process_image(image_tile,
                                                    None,
                                                    (network_lost, network_downsamples, interpolation_lost))

                if self._postprocess_function:
                    padded_result = self._postprocess_function(padded_result, tile_x, tile_y)

                if msk_tile is not None:
                    if len(msk_tile.shape) == 3 and not self._soft:
                        msk_tile = np.squeeze(msk_tile)
                    padded_result *= (msk_tile > 0)

                if len(padded_result.shape) == 3:
                    clipped_result = padded_result[crop_top:padded_result.shape[0] - crop_bottom,
                                                   crop_left:padded_result.shape[1] - crop_right, :]
                else:
                    clipped_result = padded_result[crop_top:padded_result.shape[0] - crop_bottom,
                                                   crop_left:padded_result.shape[1] - crop_right]

                clipped_result_height = clipped_result.shape[0]
                clipped_result_width = clipped_result.shape[1]

                self._write_tile_to_result(clipped_result,
                                           tile_result_y,
                                           tile_result_x,
                                           (clipped_result_height, clipped_result_width))

        sys.stdout.write('\n')
        sys.stdout.flush()

        self._finish()

    def _write_tile_to_result(self, tile, row, col, shape=None):
        """
        Writes a result tile to the result image.

        When performing tile-by-tile analysis, tiles are written to the result
        one-by-one. This also allows writing of results to disk in derived
        classes.
        """
        if tile is None:
            tile = np.zeros(shape)
        if self._result is not None:
            if self._soft:
                if self._quantize:
                    tile = np.clip(((tile - self._quantize_min) / (self._quantize_max - self._quantize_min)) * 255,
                                   0,
                                   255).astype("ubyte")

                if self._output_class >= 0:
                    self._result[row:row + tile.shape[-2], col:col + tile.shape[-1]] = tile[self._output_class]
                else:
                    self._result[row:row + tile.shape[0], col:col + tile.shape[1]] = tile
            else:
                self._result[row:row + tile.shape[-2], col:col + tile.shape[-1]] = tile
