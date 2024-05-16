"""
U-Net implementation using Keras.
"""

from . import kerasmodelbase as dmskerasmodelbase
from ...errors import modelerrors as dmsmodelerrors

from keras import models, layers, optimizers, regularizers
import keras.backend as K
import tensorflow as tf

import numpy as np

#----------------------------------------------------------------------------------------------------

class UNet(dmskerasmodelbase.KerasModelBase):
    """U-Net model implementation in Keras."""

    def __init__(self, name=None, description=None):
        """
        Initialize model parameters.

        Args:
            name (str): Name of the model.
            description (str): Description of the model.
        """

        # Initialize base class.
        #
        super().__init__(name, description)

        # Initialize members.
        #
        self.__input_shape = None                # Input shape: (rows, cols, channels).
        self.__output_shape = None               # Output shape: (rows, cols, channels
        self.__depth = 4                         # Depth of the network
        self.__branching_factor = 0              # Filter count branching factor.
        self.__classes = 0                       # Number of output classes.
        self.__batch_norm = False                # Batch normalization.
        self.__dropout_prob = 0.0                # Probability of dropout on the dropout layers.
        self.__dropout_count = 0                 # Number of dropout layers.
        self.__residual = False                  # Residual connections
        self.__downsampling = 'maxpool'          # Downsampling method
        self.__upsampling = 'upsampling2d'       # Upsampling method
        self.__output_offset = None              # Difference between input and output
        self.__padding = 'valid'                 # Padding mode.
        self.__l2_lambda = 0.0                   # L2 regularizer
        self.__loss = 'categorical_crossentropy' # Loss function

    def configure(self, input_shape, depth, classes, branching_factor, batch_norm, dropout_count, dropout_prob, l2_lambda, padding, residual, downsampling, upsampling, channels_first, loss):
        """
        Save the network configuration parameters.

        Args:
            input_shape (tuple): Number of channels and size of the individual input images. (channels, rows, cols)
            depth (int): Depth of the U-Net.
            classes (int): Number of output classes.
            branching_factor (int): Branching factor. The number of filters on the first layer is 2 on the power of branching factor.
            batch_norm (bool): Use batch normalization.
            dropout_count (int): Number of dropout layers.
            dropout_prob (float): Dropout probability.
            l2_lambda (float): L2 loss lambda.
            padding (string): Type of padding to use.
            residual (bool): Use residual connections.
            downsampling (string): One of 'maxpool' or 'strided'
            upsampling (string): One of 'transposedconv' or 'upsampling2d'
            channels_first (bool): If true, the network expect patches shaped BCHW, otherwise BHWC.
            loss (string): 'lovasz', 'categorical_crossentropy' or 'dice'.

        Raises:
            InvalidInputShapeError: The input shape is not valid.
            InvalidModelClassCountError: The number of classes is not valid.
            InvalidBranchingFactorError: The branching factor is not valid.
            InvalidDropoutLayerCountError: The number of dropout layers is not valid.
            InvalidDropoutProbabilityError: The dropout probability is out of (0.0, 1.0) bounds.
            NotSupportedChannelOrder: The dimension order is not supported.
            NotSupportedLoss: The loss is not supported.
            InvalidNumberOfClassesForLoss: The loss function doesn't support this amount of classes.
        """

        # Check input parameters.
        #
        if len(input_shape) != 3 or min(input_shape) <= 0:
            raise dmsmodelerrors.InvalidInputShapeError(input_shape)

        if classes <= 1:
            raise dmsmodelerrors.InvalidModelClassCountError(classes)

        if branching_factor < 0:
            raise dmsmodelerrors.InvalidBranchingFactorError(branching_factor)

        if dropout_count < 0 or depth < dropout_count:
            raise dmsmodelerrors.InvalidDropoutLayerCountError(dropout_count)

        if dropout_count > 0 and (dropout_prob <= 0.0 or 1.0 <= dropout_prob):
            raise dmsmodelerrors.InvalidDropoutProbabilityError(dropout_prob)

        if padding not in ['same', 'valid']:
            raise dmsmodelerrors.NotSupportedPaddingError(padding, 'conv2d')

        if channels_first:
            raise dmsmodelerrors.NotSupportedChannelOrder('channels first')

        if loss not in ['categorical_crossentropy', 'lovasz', 'dice']:
            raise dmsmodelerrors.NotSupportedLoss(loss)

        if loss == 'lovasz' and classes <= 2:
            raise dmsmodelerrors.InvalidNumberOfClassesForLoss(loss, classes)

        # Save parameters.
        #
        self.__input_shape = input_shape
        self.__output_shape = None
        self.__depth = depth
        self.__classes = classes
        self.__branching_factor = branching_factor
        self.__batch_norm = batch_norm
        self.__dropout_count = dropout_count
        self.__dropout_prob = dropout_prob
        self.__residual = residual
        self.__downsampling = downsampling
        self.__upsampling = upsampling
        self.__padding = padding
        self.__l2_lambda = l2_lambda
        self.__loss = loss

    def _customdata(self):
        """
        Get a map of custom data structure to add to the dictionary of exported values.

        Returns:
            dict: Dictionary of values to add to the exported data.
        """

        # Collect custom data.
        #
        return {'depth': self.__depth,
                'classes': self.__classes,
                'branching_factor': self.__branching_factor,
                'batch_norm': self.__batch_norm,
                'dropout_count': self.__dropout_count,
                'dropout_prob': self.__dropout_prob,
                'l2_lambda': self.__l2_lambda,
                'padding': self.__padding,
                'upsampling': self.__upsampling,
                'downsampling': self.__downsampling,
                'residual': self.__residual,
                'loss': self.__loss}

    def _setcustomdata(self, data_maps):
        """
        Set the class specific data.

        Args:
            data_maps (dict): Custom data map.
        """

        # Configure custom data.
        #
        self.__depth = data_maps.get('depth', 0)
        self.__classes = data_maps.get('classes', 0)
        self.__branching_factor = data_maps.get('branching_factor', 0)
        self.__batch_norm = data_maps.get('batch_norm', False)
        self.__dropout_count = data_maps.get('dropout_count', 0)
        self.__dropout_prob = data_maps.get('dropout_prob', 0.0)
        self.__l2_lambda = data_maps.get('l2_lambda', 0.0)
        self.__padding = data_maps.get('padding', None)
        self.__downsampling = data_maps.get('downsampling', 'maxpool')
        self.__upsampling = data_maps.get('upsampling', 'upsampling2d')
        self.__loss = data_maps.get('loss', 'categorical_crossentropy')

    def build(self):
        """Build the network instance with the pre-configured parameters."""

        inputs, outputs = self.__networkdefinition()

        self._model_instance = models.Model(inputs, outputs)
        optimizer = optimizers.Adam(lr=0.0001, beta_1=0.99, beta_2=0.99)

        if self.__loss == 'categorical_crossentropy':
            self._model_instance.compile(optimizer=optimizer, loss=['categorical_crossentropy'], sample_weight_mode="temporal", metrics=['accuracy',
                                                                                                                                         self.__dice_coef,
                                                                                                                                         self.__iou_coef])

        if self.__loss == 'lovasz':
            self._model_instance.compile(optimizer=optimizer, loss=[self.__keras_lovasz_softmax], sample_weight_mode="temporal", metrics=['accuracy',
                                                                                                                                           self.__dice_coef,
                                                                                                                                           self.__iou_coef])

        if self.__loss == 'dice':
              self._model_instance.compile(optimizer=optimizer, loss=[self.__dice_coef_loss], sample_weight_mode="temporal", metrics=['accuracy',
                                                                                                                                     self.__dice_coef,
                                                                                                                                     self.__iou_coef])

        self._compiled = True

        # Retrieve output shape of the intermediate output layer to reshape predictions.
        #
        self.__output_shape = self._model_instance.get_layer('output').output_shape

        self._model_instance.metrics_tensors += self._model_instance.outputs
        self._model_instance.metrics_names += ['predictions']

        self._addcustommetric(name='errors')

    def _restoremodelparameters(self, parameters):
        """
        Custom restore function. Makes sure that the output tensors are correctly set, even after a load.

        Args:
            parameters (dict): Parameters of the model.
        """

        # Load custom loss functions, otherwise keras will crash.
        #
        from keras.utils.generic_utils import get_custom_objects
        custom_lovasz_loss = self.__keras_lovasz_softmax
        custom_dice_loss = self.__dice_coef_loss
        custom_dice_coef = self.__dice_coef
        custom_iou_coef = self.__iou_coef

        get_custom_objects().update({'__keras_lovasz_softmax': custom_lovasz_loss,
                                     '__dice_coef_loss': custom_dice_loss,
                                     '__dice_coef': custom_dice_coef,
                                     '__iou_coef': custom_iou_coef})

        super()._restoremodelparameters(parameters)

        # Explicitly reset the train and validate functions of the internal model
        # TODO this is a workaround for https://github.com/fchollet/keras/issues/8468
        #
        self._model_instance.train_function = None
        self._model_instance.test_function = None

        self._model_instance.metrics_tensors += self._model_instance.outputs
        self._model_instance.metrics_names += ['predictions']

        # Set output shape
        #
        self.__output_shape = self._model_instance.get_layer('output').output_shape

    def update(self, x, y, sample_weight=None, class_weight=None, *args, **kwargs):
        """
        Update the network.

        Args:
            x (numpy.ndarray): an array containing image data.
            y (numpy.ndarray ): an array containing label data.
            sample_weights (numpy.ndarray): an array containing weights for samples to increase their contribution to the loss
            class_weights (numpy.ndarray): an array containing weights for classes to increase their contribution to the loss

        Returns:
            dict: Output of the update function.
        """

        # Match labels and weights to the output of the network. Apply cropping if required.
        #
        sample_weight = self.__matchdatatonetwork(sample_weight)
        y = self.__matchdatatonetwork(y)

        # Flatten the weights and y as Keras does not support 3D sample weight maps
        #
        if sample_weight is not None:
            sample_weight = sample_weight.reshape(sample_weight.shape[0],
                                                  sample_weight.shape[1] * sample_weight.shape[2])
        y = y.reshape(y.shape[0], y.shape[1] * y.shape[2], y.shape[3])

        if self.__loss == 'lovasz' and y.shape[-1] == self.__classes:
            # The implementation of the lovasz loss expects a tensor with the shape [B, H, W]. Therefore I take the argmax if one-hot.
            # Unfortunaly, I can't check if one-hot is used to do this inside the loss function itself.
            # TODO: check if we can figure out if one-hot is used or not.

            y = np.expand_dims(np.argmax(y, axis=-1), axis=-1)

        output = super().update(x, y, sample_weight, class_weight, *args, **kwargs)

        #calculate error
        #
        weight_sum = np.sum(sample_weight, axis=-1).reshape(sample_weight.shape[0], 1)
        error_flat = (np.max((y - output['predictions']), axis=-1) * sample_weight) / weight_sum
        output['errors'] = np.clip(np.sum(error_flat, axis=-1), 0, 1)

        # Reshape the predictions.
        #
        output['predictions'] = output['predictions'].reshape((-1,) + self.__output_shape[1:])

        return output

    def validate(self, x, y, sample_weight=None, *args, **kwargs):
        """
        Validate the network.

        Args:
            x (numpy.ndarray or list of numpy.ndarray): contains image data.
            y (numpy.ndarray or list of numpy.ndarray ): contains label data.
            sample_weights (numpy.ndarray): an array containing weights for samples to increase their contribution to the loss

        Returns:
            dict: Output of the validation function.
        """

        # Match labels and weights to the output of the network. Apply cropping if required.
        #
        sample_weight = self.__matchdatatonetwork(sample_weight)
        y = self.__matchdatatonetwork(y)

        # Flatten the weights as Keras does not support 3D sample weight maps
        #
        if sample_weight is not None:
            sample_weight = sample_weight.reshape(sample_weight.shape[0],
                                                  sample_weight.shape[1] * sample_weight.shape[2])
        y = y.reshape(y.shape[0], y.shape[1] * y.shape[2], y.shape[3])
        
        if self.__loss == 'lovasz' and y.shape[-1] == self.__classes:
            # The implementation of the lovasz loss expects a tensor with the shape [B, H, W]. Therefore I take the argmax if one-hot.
            # Unfortunaly, I can't check if one-hot is used to do this inside the loss function itself.
            # TODO: check if we can figure out if one-hot is used or not.

            y = np.expand_dims(np.argmax(y, axis=-1), axis=-1)   

        output = super().validate(x, y, sample_weight, *args, **kwargs)

        # Reshape the predictions.
        #
        output['predictions'] = output['predictions'].reshape((-1,) + self.__output_shape[1:])

        return output

    def predict(self, x, *args, **kwargs):
        """
        Use the network for evaluation.

        Args:
            x (numpy.ndarray or list of numpy.ndarray): contains image data.

        Returns:
            dict: Output of the evaluation function.
        """

        # Run the model on the set of patches.
        #
        output = super().predict(x, *args, *kwargs)

        # Reshape the predictions.
        #
        output['predictions'] = [prediction.reshape((-1,) + self.outputshape[1:]) for prediction in output['predictions']]

        return output

    @property
    def inputshape(self):
        """
        Return input shape of unet

        Returns:
            list: input shape
        """

        if not self._compiled:
            raise dmsmodelerrors.ModelNotCompiledError()

        if not self.__input_shape:
            self.__input_shape = self._model_instance.input_shape

        return self.__input_shape

    @property
    def outputshape(self):
        """
        Return output shape of unet

        Returns:
            list: Output shape

        Raises:
            ModelNotCompiledError
        """

        if not self._compiled:
            raise dmsmodelerrors.ModelNotCompiledError()

        if not self.__output_shape:
            self.__output_shape = self._model_instance.get_layer('output').output_shape

        return self.__output_shape

    def __conv_block(self, input, filters, activation, l2, dropout_enabled):
        """
        Build a single convolution block of the U-Net.

        Args:
            input: Network input at this point.
            filters (int): Number of filters.
            activation (string|callable): Activation function for the convolution layers.
            l2 (Regularizer): Regularizer object
            dropout_enabled (bool): Whether to add dropout in this layer
        """

        net = layers.Conv2D(filters, 3, activation=activation, padding=self.__padding, kernel_regularizer=l2)(input)

        if self.__batch_norm:
            net = layers.BatchNormalization()(net)

        if dropout_enabled and self.__dropout_prob > 0:
            # Dropout per channel.
            #
            noise_shape = (None, 1, 1, filters)

            net = layers.Dropout(self.__dropout_prob, noise_shape=noise_shape)(net)

        net = layers.Conv2D(filters, 3, activation=activation, padding=self.__padding, kernel_regularizer=l2)(net)

        if self.__batch_norm:
            net = layers.BatchNormalization()(net)

        if self.__residual:
            total_crop = K.int_shape(input)[1] - K.int_shape(net)[1]
            halve_crop = total_crop // 2

            cropped_input = layers.Cropping2D(
                ((halve_crop, total_crop - halve_crop), (halve_crop, total_crop - halve_crop)))(input)

            net = layers.Concatenate()([cropped_input, net])

        return net

    def __level_block(self, net, filters, depth, activation, l2):
        """
        Recursively build the U-Net.

        Args:
            net: Current network object.
            filters (int): Number of filters on this level.
            depth (int): Depth at this stage.
            activation (string|callable): Activation function for layers.
            l2 (Regularizer): Regularizer object
        """

        dropout_enabled = self.__dropout_count >= depth

        if depth > 1:
            first_block = self.__conv_block(input=net, filters=filters, activation=activation, l2=l2, dropout_enabled=dropout_enabled)

            if self.__downsampling == 'maxpool':
                net = layers.MaxPooling2D(pool_size=2, strides=2)(first_block)

            else:
                net = layers.Conv2D(filters, 3, strides=2, padding=self.__padding, kernel_regularizer=l2)(first_block)

            # Recursively go a level deeper.
            #
            net = self.__level_block(net=net, filters=2 * filters, depth=depth - 1, activation=activation, l2=l2)

            if self.__upsampling == 'transposedconv':
                net = layers.Conv2DTranspose(filters, 2, strides=2, activation=activation, padding=self.__padding, kernel_regularizer=l2)(net)
            else:
                net = layers.UpSampling2D()(net)
                net = layers.Conv2D(filters, 2, activation=activation, padding=self.__padding, kernel_regularizer=l2)(net)

            # We use valid padding so make sure the two blocks are the same size. Cropping is done from the center with
            # any remainder (in case of uneven crop) will be cropped from the bottom and right.
            #
            total_crop = K.int_shape(first_block)[1] - K.int_shape(net)[1]
            halve_crop = total_crop // 2

            cropped_block = layers.Cropping2D(((halve_crop, total_crop - halve_crop), (halve_crop, total_crop - halve_crop)))(first_block)

            # Connect two paths of the U-Net.
            #
            net = layers.Concatenate()([cropped_block, net])

            # Add last conv block.
            #
            net = self.__conv_block(input=net, filters=filters, activation=activation, l2=l2, dropout_enabled=dropout_enabled)
        else:
            net = self.__conv_block(input=net, filters=filters, activation=activation, l2=l2, dropout_enabled=dropout_enabled)

        return net

    def __networkdefinition(self):
        """
        Network definition.

        Returns
            Tensor: The network architecture.
        """

        input = layers.Input(shape=self.__input_shape, name='input')
        l2 = regularizers.l2(self.__l2_lambda)

        net = self.__level_block(input, 2 ** self.__branching_factor, self.__depth, 'relu', l2)

        output = layers.Conv2D(self.__classes, 1, activation='softmax', name='output', kernel_regularizer=l2)(net)

        # Flatten the output to (width * height, n_classes) as Keras does not support 3D sample weight maps
        #
        output_shape = K.int_shape(output)
        flatten = layers.Reshape((output_shape[1] * output_shape[2], output_shape[3]))(output)

        return input, flatten

    def __matchdatatonetwork(self, data):
        """
        Slice label or weight patches based on the output size of the network. Values on the edges are discarded. Does nothing if the output size matches the label size.

        Args:
            data (np.array): A numpy array containing the labels or weights

        Returns:
            np.array: The sliced labels
        """

        # No need to reshape if the shape of the labels matches the output shape.
        #
        if self.__output_shape[1:3] == data.shape[1:3]:
            return data

        # Compute the offset based on the difference between input and output.
        #
        if self.__output_offset is None:
            # Returns an array with offset for [left, right, top, bottom].
            #
            self.__output_offset, _, _ = self.getreconstructioninformation()

        # Slice the labels based on the offset.
        #
        return data[:, self.__output_offset[0]:-self.__output_offset[1], self.__output_offset[2]:-self.__output_offset[3], :]

    # Add metrics
    #
    def __iou_coef(self, y_true, y_pred):
        ''''
        Intersection over Union metric
        '''
        intersection = y_true * y_pred
        union = y_true + ((1 - y_true) * y_pred)

        return K.sum(intersection, axis=-1) / (K.sum(union, axis=-1) + K.epsilon())

    def __dice_coef(self, y_true, y_pred):
        '''
        Dice coefficient metric
        '''

        intersect = K.sum(y_true * y_pred, axis=-1)
        denominator = K.sum(y_true + y_pred, axis=-1)

        return K.mean((2. * intersect / (denominator + K.epsilon())))

    # Code taken from https://github.com/bermanmaxim/LovaszSoftmax. Adapted to work with Keras and our library
    #
    def __lovasz_grad(self, gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper
        """
        gts = tf.reduce_sum(gt_sorted)
        intersection = gts - tf.cumsum(gt_sorted)
        union = gts + tf.cumsum(1. - gt_sorted)
        jaccard = 1. - intersection / union
        jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
        return jaccard

    def __lovasz_softmax(self, y_true, y_pred):
        """
        Multi-class Lovasz-Softmax loss
          probas: [B, H, W, C] Tensor, class probabilities at each prediction (between 0 and 1)
          labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
        """

        # Reshape the inputs
        #
        probas = tf.reshape(y_pred, (-1, self.__classes))
        labels = tf.reshape(y_true, (-1,))

        # Calculate the intersection
        #
        loss = self.__lovasz_softmax_flat(probas, labels, classes='present')

        return loss

    def __lovasz_softmax_flat(self, probas, labels, classes='all'):
        """
        Multi-class Lovasz-Softmax loss
          probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
          labels: [P] Tensor, ground truth labels (between 0 and C - 1)
          classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        """
        C = self.__classes
        losses = []
        present = []

        # TODO: Check if we ever want to have the option to calculate the loss per patch.
        #
        class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
        for c in class_to_sum:
            fg = tf.cast(tf.equal(labels, c), probas.dtype)  # foreground for class c
            if classes == 'present':
                present.append(tf.reduce_sum(fg) > 0)
            errors = tf.abs(fg - probas[:, c])
            errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort_{}".format(c))
            fg_sorted = tf.gather(fg, perm)
            grad = self.__lovasz_grad(fg_sorted)
            losses.append(
                tf.tensordot(errors_sorted, tf.stop_gradient(grad), 1, name="loss_class_{}".format(c))
            )
        if len(class_to_sum) == 1:  # short-circuit mean when only one class
            return losses[0]
        losses_tensor = tf.stack(losses)
        if classes == 'present':
            present = tf.stack(present)
            losses_tensor = tf.boolean_mask(losses_tensor, present)
        loss = tf.reduce_mean(losses_tensor)
        return loss

    def __keras_lovasz_softmax(self, y_true, y_pred):
        ''''
        Multi class Lovasz loss to minimize.
        '''
        return self.__lovasz_softmax(y_true, y_pred)

    def __dice_coef_loss(self, y_true, y_pred):
        '''
        Dice loss to minimize.
        '''
        return 1 - self.__dice_coef(y_true, y_pred)
