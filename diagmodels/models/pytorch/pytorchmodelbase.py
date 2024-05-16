
"""
Base class for network models.
"""

from .. import modelbase as dmsmodelbase
from ...errors import modelerrors as dmsmodelerrors
# from ...utils import network as dmsnetwork

import torch
import torch.nn.functional as F

import numpy as np

#----------------------------------------------------------------------------------------------------

class PytorchModelBase(dmsmodelbase.ModelBase):
    """This class is the base class for all network model classes implemented using Keras."""

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
        self.__custom_metrics = set()  # Custom metric names.
        self.__output_offset = None
        self.__layers = None
        self.__is_training = False
        self.__is_eval = False
        self._optimizer = None
        self._loss = None

    def updatelearningrate(self, learning_rate):
        """
        Update the learning rate.

        Args:
            learning_rate (float): New learning rate.
        """

        if self._model_instance and self._optimizer:
            for param_group in self._optimizer.param_groups:
                param_group['lr'] = learning_rate

    def _addcustommetric(self, name):
        """
        Add a custom metric name to the list of available metrics.

        Args:
            name (str): Name of the metric.
        """

        self.__custom_metrics.add(name)

    def metricnames(self):
        """
        Get the list of metric names that the network returns.

        Returns:
            list: Metric names.
        """

        return ['accuracy', 'loss', 'errors']

    def losscriterion(self, logits, targets, sample_weight=None, class_weight=None, *args, **kwargs):
        targets = torch.max(targets, 1)[1]
        # TODO: I use torch.max(targets, 1)[1] here because i wasn't aware we could turn off one_hot labels
        # change this to targets when label mode is not one_hot!!
        # be aware this changes calcmetric too
        if sample_weight is not None:
            crit = torch.nn.NLLLoss(reduction='none', weight=class_weight)
            sample_weight = sample_weight.cuda().squeeze(1)
            total_loss = 0
            logits = F.log_softmax(logits, dim=1)
            per_sample_loss = crit(logits, targets.squeeze(1))
            for i, loss in enumerate(per_sample_loss):
                sample_loss = loss
                sample_loss *= sample_weight[i]
                total_loss += sample_loss[sample_weight[i] > 0].mean()
            total_loss /= len(sample_weight)
        else:
            logits = F.log_softmax(logits, dim=1)
            crit = torch.nn.NLLLoss(weight=class_weight)
            total_loss = crit(logits, targets)
        return total_loss

    def _modelparameters(self):
        """
        Collect the parameters of the network to save.

        Returns:
            dict: Dictionary of parameters.
        """

        if self._model_instance:
            state = {
                'model': self._model_instance.state_dict(),
                'optimizer': self._optimizer.state_dict(),
            }
            return state
        else:
            return None

    def _restoremodelparameters(self, parameters):
        """
        Restores the state of the model and the optimizer

        Args:
            parameters: Dictionary of parameters
        """

        # Load the state dict
        #
        if not self._model_instance:
            self.build()

        if self._model_instance:
            self._model_instance.load_state_dict(parameters['model'])

        if self._optimizer:
            self._optimizer.load_state_dict(parameters['optimizer'])

    def setchannelsorder(self, channel_first):
        """
        Set the channels first dimension order for the network.

        Note! This is a glob setting, meaning if two instances are initiated simultaneous they should have the same data format.

        Args:
            channel_first (bool): Channels should be the first after the batch dimension.
        """
        if not channel_first:
            print('Pytorch only support channels first!')
            # TODO: make actual error

    def _customdata(self):
        """
        Add data format ('channels first' or 'channels last') to model.

        Returns:
            dict: Dictionary of values to add to the exported data.
        """

        # By default it returns an empty map.
        #
        return {}

    @property
    def dimensionorder(self):
        """
        Get the dimension order.

        Returns:
            str: Channel order descriptor.
        """

        return 'bchw'

    def calcmetric(self, name, logit, target, sample_weight):
        logit = F.softmax(logit, dim=1)
        preds = torch.argmax(logit, dim=1).cpu()
        y_class = torch.argmax(target, dim=1).cpu()
        if name == 'accuracy':
            if sample_weight is not None:
                metric = (preds[sample_weight.squeeze(1) > 0] == y_class[sample_weight.squeeze(1) > 0]).sum()
                metric /= (sample_weight.squeeze(1) > 0).sum()
            else:
                metric = (preds == y_class).sum() / preds.numel()
        elif name == 'errors':
            weight_sum = np.sum(sample_weight.numpy(), axis=(1, 2, 3)).reshape(sample_weight.shape[0], 1, 1)
            error_flat = torch.max((target - logit), dim=1)[0].cpu() * sample_weight.squeeze(1)
            error_flat = error_flat.detach().numpy()
            error_flat /= weight_sum
            metric = np.sum(error_flat, axis=(1, 2))
        return metric

    def update(self, x, y, sample_weight=None, class_weight=None, *args, **kwargs):
        """
        Update the network.

        Args:
            x (numpy.ndarray): an array containing image data.
            y (numpy.ndarray ): an array containing label data.
            sample_weight (numpy.ndarray): an array containing weights for samples to increase their contribution to the loss
            class_weight (numpy.ndarray): an array containing weights for classes to increase their contribution to the loss

        Returns:
            dict: Output of the update function.
        """

        # Check if network is in training mode (important for batchnorm / dropout layers)
        #
        if not self.__is_training:
            torch.set_grad_enabled(True)
            self._model_instance.train()
            self.__is_training = True
            self.__is_eval = False

        # We need to crop output if network outputs smaller segmentation map
        #
        if self.__output_offset is None:
            self.__output_offset, _, _ = self.getreconstructioninformation(x.shape)

        sample_weight = self.__matchdatatonetwork(sample_weight)
        y = self.__matchdatatonetwork(y)

        # Forward pass
        #
        self._optimizer.zero_grad()
        x, y = torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()
        output_values = self._model_instance(x)

        # Calculate loss
        #
        sample_weight = torch.from_numpy(sample_weight) if sample_weight is not None else None
        class_weight = torch.from_numpy(class_weight) if class_weight is not None else None
        loss = self.losscriterion(output_values, y.long(), sample_weight, class_weight)

        # Backward pass
        #
        loss.backward()
        self._optimizer.step()

        # Create metric dictionary
        #
        output = {}
        metrics = self.metricnames()

        for name in metrics:
            if name == 'loss':
                metric = float(loss.detach().cpu())
            else:
                metric = self.calcmetric(name, output_values, y, sample_weight)
            output[name] = metric

        return output

    def __matchdatatonetwork(self, data):
        """
        Slice label or weight patches based on the output size of the network. Values on the edges are discarded. Does nothing if the output size matches the label size.

        Args:
            data (np.array): A numpy array containing the labels or weights

        Returns:
            np.array: The sliced labels
        """

        # Compute the offset based on the difference between input and output.
        #
        if self.__output_offset is None:
            # Returns an array with offset for [left, right, top, bottom].
            # Since Pytorch is channels first we need to transpose
            #
            self.__output_offset, _, _ = self.getreconstructioninformation((data.shape[2], data.shape[3], data.shape[1]))

        # Slice the labels based on the offset.
        #
        return data[:, :,
                    self.__output_offset[0]:data.shape[2] - self.__output_offset[1],
                    self.__output_offset[2]:data.shape[3] - self.__output_offset[3]]

    def validate(self, x, y, sample_weight=None, class_weight=None, *args, **kwargs):
        """
        Validate the network.

        Args:
            x (numpy.ndarray or list of numpy.ndarray): contains image data.
            y (numpy.ndarray or list of numpy.ndarray): contains label data.
            sample_weight (numpy.ndarray): an array containing weights for samples to increase their contribution to the loss

        Returns:
            dict: Output of the validation function.
        """

        # Turn off gradient calculation, put all modules in eval mode:
        #
        if not self.__is_eval:
            torch.set_grad_enabled(False)
            self._model_instance.eval()
            self.__is_eval = True
            self.__is_training = False

        # We need to crop output if network outputs smaller segmentation map
        #
        sample_weight = self.__matchdatatonetwork(sample_weight)
        y = self.__matchdatatonetwork(y)

        # Forward pass
        #
        x, y = torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()
        output_values = self._model_instance(x)

        # Calculate loss
        #
        sample_weight = torch.from_numpy(sample_weight) if sample_weight is not None else None
        class_weight = torch.from_numpy(class_weight) if class_weight is not None else None
        loss = self.losscriterion(output_values, y.long(), sample_weight, class_weight)

        # Create metric dictionary
        #
        output = {}
        metrics = self.metricnames()

        for name in metrics:
            if name == 'loss':
                metric = float(loss.detach().cpu())
            else:
                metric = self.calcmetric(name, output_values, y, sample_weight)
            output[name] = metric
        return output

    def predict(self, x, *args, **kwargs):
        """
        Use the network for evaluation.

        Args:
            x (numpy.ndarray or list of numpy.ndarray): contains image data.

        Returns:
            dict: Output of the evaluation function.
        """

        # Calculate reconstruction information if needed
        #
        if self.__output_offset is None:
            self.__output_offset, _, _ = self.getreconstructioninformation(x.shape)

        # Turn model into eval mode
        #
        if not self.__is_eval:
            torch.set_grad_enabled(False)
            self._model_instance.eval()
            self.__is_eval = True
            self.__is_training = False

        # Do forward pass
        #
        # Library will always give channels last to predict function
        #
        x = torch.from_numpy(x.transpose(0, 3, 1, 2)).cuda().float()
        pred = self._model_instance(x)
        pred = F.softmax(pred, dim=1)
        pred = pred.detach().cpu()

        # Library expects predictions in channels last format
        #
        pred = pred.numpy().transpose(0, 2, 3, 1)
        return {"predictions": pred}

    def getreconstructioninformation(self, input_shape=None):
        """
        Calculate the scale factor and padding to reconstruct the input shape.

        This function calculates the information needed to reconstruct the input image shape given the output of a layer. For each layer leading up to the
        output it will return the number of pixels lost/gained on all image edges.

        For transposed convolutions the function checks the stride and cropping method to compute the correct upsampling factor.

        Args:
            input_shape (sequence of ints): input_shape to calculate the reconstruction information for. Order should be (nr_channels, width, height)
                set of layers.

        Returns:
            np.array: lost pixels
            np.array: downsample factor
            np.array: interpolation factor

        Raises:
            MissingNetworkError: The network is not defined.
        """

        # PyTorch tactic:
        # 1. Add hooks to all modules
        # 2. Do forward pass with fake data
        # 3. Follow linear path back from prediction via grad_fn
        # 4. Calculate recon-information with this linear path
        #

        if self._model_instance is None:
            raise dmsmodelerrors.MissingNetworkError()

        # 1. Add hooks to all relevant modules
        #
        def forw_lambda(module, inpt, outpt):
            self._forwardstatshook(module, inpt, outpt)

        hooks = []

        supported_layers = (torch.nn.Conv2d,
                            torch.nn.MaxPool2d,
                            torch.nn.AvgPool2d,
                            torch.nn.AdaptiveAvgPool2d,
                            torch.nn.AdaptiveMaxPool2d,
                            torch.nn.ConvTranspose2d,
                            torch.nn.UpsamplingBilinear2d,
                            torch.nn.UpsamplingNearest2d)

        if self.__layers is None:
            self.__layers = self._get_layers(self._model_instance.children())

        for mod in self.__layers:
            if isinstance(mod, supported_layers):
                forw_handle = mod.register_forward_hook(forw_lambda)
                hooks.append(forw_handle)

        # Create a dictionary where we keep track of gradient function -> module
        #
        self._grad_fn_to_module = {}

        # Create fake data to do the forward pass
        #
        fake_data = torch.cuda.FloatTensor(2, 3, input_shape[0], input_shape[1]).fill_(1)

        # Do forward pass
        #
        torch.set_grad_enabled(True)
        output = self._model_instance(fake_data)

        # Find linear path
        # We walk the backward function (grad_fn / next_functions) until the end.
        # We gather all the modules we recognize from the forward pass
        # (The ones that have been saved under their respective grad_fn in the hook.)
        #
        prev = output.grad_fn
        modules = []
        while True:
            if prev in self._grad_fn_to_module:
                mod = self._grad_fn_to_module[prev]
                modules.append(mod)
            if hasattr(prev, 'next_functions') and len(prev.next_functions) > 0:
                prev = prev.next_functions[0][0]
            else:
                break

        # Clean memory
        #
        del self._grad_fn_to_module
        for hook in hooks:
            hook.remove()

        # Do actual reconstruction information calculation with the linear path
        #
        return self._getreconstructioninformationforlayers(input_shape, modules[::-1])

    def _get_layers(self, modules=None):
        """
        Recursive function to find all the modules of a PyTorch network
        """
        layers = []
        for mod in modules:
            layers.append(mod)
            try:
                mod_layers = self._get_layers(mod.children())
                layers.extend(mod_layers)
            except StopIteration:
                pass

        return layers

    def _forwardstatshook(self, module, input, output):
        """
        Save the module that is coupled to the grad_fn of the tensor.
        Somehow this cannot be retrieved during the backward pass.
        """
        self._grad_fn_to_module[output.grad_fn] = module

    def getnumberofoutputchannels(self):
        """
        Returns the output channels of the network
        """
        if self._model_instance:
            if self.__layers is None:
                self.__layers = self._get_layers(self._model_instance.children())
            for mod in reversed(self.__layers):
                if hasattr(mod, 'out_channels'):
                    return int(mod.out_channels)
        else:
            raise dmsmodelerrors.MissingNetworkError()

    def _getreconstructioninformationforlayers(self, input_shape, layers):
        lost = []
        downsamples = []
        last_shape = torch.FloatTensor([input_shape[0], input_shape[1]])

        for l in layers:
            if isinstance(l, (torch.nn.AdaptiveAvgPool2d, torch.nn.AdaptiveMaxPool2d)):
                next_shape = torch.FloatTensor([1, 1])
                downsamples.append([l.kernel_size[0], l.kernel_size[0]])
            elif isinstance(l, torch.nn.ConvTranspose2d):
                # Currently only valid padding with a filter_size of 2 and stride 2 is supported
                #
                if l.kernel_size != (2, 2):
                    raise dmsmodelerrors.NotSupportedLayerConfigurationError('filter_size', str(l.kernel_size), str(l))
                elif l.strides != (2, 2):
                    raise dmsmodelerrors.NotSupportedLayerConfigurationError('stride', str(l.strides), str(l))

                # A valid padding with filter size of 2 and stride of 2 results in a output size
                # that is double the size of the input image. No pixels are lost.
                next_shape = last_shape * 2.0
                downsamples.append(torch.FloatTensor([0.5, 0.5]))
            elif isinstance(l, (torch.nn.UpsamplingBilinear2d, torch.nn.UpsamplingNearest2d)):
                upsample = 1 / l.scale_factor
                next_shape = last_shape * upsample
                downsamples.append(torch.FloatTensor([upsample, upsample]))
            else:
                padding = [l.padding] * 2 if not isinstance(l.padding, tuple) else l.padding
                cur_stride = [l.stride] * 2 if not isinstance(l.stride, tuple) else l.stride
                kernel_size = [l.kernel_size] * 2 if not isinstance(l.kernel_size, tuple) else l.kernel_size
                dilation = [l.dilation] * 2 if not isinstance(l.dilation, tuple) else l.dilation

                cur_stride = torch.FloatTensor(cur_stride)
                kernel_size = torch.FloatTensor(kernel_size)
                padding = torch.FloatTensor(padding)
                dilation = torch.FloatTensor(dilation)

                # According to pytorch documentation
                out_shape = torch.FloatTensor((last_shape[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / cur_stride + 1)
                out_shape = torch.floor(out_shape)

                # Handle dilation
                if dilation[0] > 1:
                    kernel_size = kernel_size + (kernel_size - torch.FloatTensor([1, 1])) * (dilation - torch.FloatTensor([1, 1]))

                lost_this_layer = torch.FloatTensor([0, 0, 0, 0])  # left, right, top, bottom

                if isinstance(l, (torch.nn.Conv2d, torch.nn.MaxPool2d, torch.nn.AvgPool2d)):
                    if padding[0] == kernel_size[0] // 2:  # same padding
                        next_shape = torch.ceil(last_shape / cur_stride)
                    elif padding[0] == 0:
                        next_shape = torch.floor((last_shape - kernel_size) / cur_stride) + 1

                    cutoff = last_shape - out_shape * cur_stride

                    if padding[0] == 0:  # same padding
                        lost_this_layer[0] = (kernel_size[1] - cur_stride[1]) / 2
                        lost_this_layer[2] = (kernel_size[0] - cur_stride[0]) / 2
                        lost_this_layer[1] = (kernel_size[1] - cur_stride[1]) / 2 + cutoff[0]
                        lost_this_layer[3] = (kernel_size[0] - cur_stride[0]) / 2 + cutoff[1]
                    elif padding[0] == kernel_size[0] // 2:
                        lost_this_layer[1] = cutoff[0]
                        lost_this_layer[3] = cutoff[1]
                    else:
                        next_shape = torch.ceil(last_shape / cur_stride)
                        # TODO: test this!
                        # raise dmsmodelerrors.NotSupportedPaddingError(str(l.padding), str(l))
                downsamples.append(cur_stride)

            last_shape = next_shape
            lost.append(lost_this_layer)

        # Convert to float for potential upsampling
        #
        for i in range(1, len(downsamples)):
            downsamples[i] *= downsamples[i - 1]
            lost[i][0:2] *= downsamples[i - 1][0]
            lost[i][2:] *= downsamples[i - 1][1]

        # Sum up the lost pixels and convert to normal python int lists
        #
        lost_total = torch.stack(lost).sum(dim=0)
        lost_total[0::2] = torch.floor(lost_total[0::2])
        lost_total[1::2] = torch.ceil(lost_total[1::2])

        interpolation_lost = torch.ceil((downsamples[-1] - 1) / 2)
        interpolation_lost = np.array([interpolation_lost[0], interpolation_lost[0], interpolation_lost[1], interpolation_lost[1]])

        return lost_total.numpy().astype(np.int32), downsamples[-1].numpy(), interpolation_lost

    def unfix(self):
        """Unfix input shape."""

        pass
