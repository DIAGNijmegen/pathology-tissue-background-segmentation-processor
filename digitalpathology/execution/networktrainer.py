"""
This file contains class for training networks.
"""

from ..errors import trainingerrors as dpttrainingerrors
from ..batch import batchgenerator as dptbatchgenerator
from ..stats import stataggregator as dptstats
from ..utils import serialize as dptserialize
from ..utils import trace as dpttrace

import diagmodels.models.modelbase as dmsmodelbase

import logging
import numpy as np
import time
import os
import sys
import math
import hashlib

#----------------------------------------------------------------------------------------------------

class NetworkTrainer(object):
    """This class can train a network."""

    def __init__(self, model, training_batch_generator, validation_batch_generator, stat_aggregator, file_synchronizer, metric_name, higher_is_better, averaging_length, iter_log_percent):
        """
        Initialize the object. Save the model to train and the configured batch generators.

        Args:
            model (dmsmodelbase.ModelBase): Network to train.
            training_batch_generator (dptbatchgenerator.BatchGenerator): Training batch generator.
            validation_batch_generator (dptbatchgenerator.BatchGenerator): Validation batch generator.
            stat_aggregator (dptstats.StatAggregator): Statistics aggregator.
            file_synchronizer (dptfilesynchronizer.FileSynchronizer): File synchronizer object.
            metric_name (str): Name of the metric to evaluate the performance of the network.
            higher_is_better (bool): Higher metric value is better.
            averaging_length (int): Number of epochs to average over to get the actual metric.
            iter_log_percent (float): An iteration log entry is made before every iter_log_percent chunk of iterations.

        Raises:
            InvalidIterationLogPercentError: The iteration log percent is out of (0.0, 1.0] bounds.
            InvalidNetworkObjectError: The network object is invalid.
            InvalidBatchGeneratorError: The batch generator object is invalid.
            DimensionOrderMismatchError: Model - generator dimension order mismatch.
            InvalidFileSynchronizerError: The file synchronizer is not valid.
            InvalidStatsHandlerError: The statistics handler is invalid.
            InvalidAveragingLengthError: The epoch averaging length is not valid.
        """

        # Initialize the base class.
        #
        super().__init__()

        # Initialize members.
        #
        self.__model = None            # Model to train.
        self.__training_gen = None     # Training batch generator.
        self.__validation_gen = None   # Validation batch generator.
        self.__syncer = None           # File synchronizer.
        self.__stats = None            # Stats aggregator.
        self.__iter_log_percent = 1.0  # Log tick frequency for iteration progress reporting.
        self.__logger = None           # Configured logger object.

        self.__plateau_metric_name = None              # Name of the metric used to measure model improvements.
        self.__plateau_averaged_metric_name = None     # Name of the metric used to measure model improvements that is averaged over the given epochs.
        self.__plateau_metric_higher_is_better = True  # Whether a higher value of the plateau metric is better than lower.
        self.__plateau_epoch_averaging_length = 0      # Number of epochs to average over when determining plateau.
        self.__plateau_metric_last_values = []         # Holds the last X values for the plateau metric.
        self.__plateau_metric_over_epochs = []         # List in which the metric value per epoch is stored to determine plateau.

        # Save parameters.
        #
        self.__initlogging(iter_log_percent=iter_log_percent)
        self.__setmodel(model=model)
        self.__setgenerators(training_batch_generator=training_batch_generator, validation_batch_generator=validation_batch_generator)
        self.__setmetric(metric_name=metric_name, higher_is_better=higher_is_better, averaging_length=averaging_length)
        self.__setsyncer(file_synchronizer=file_synchronizer)
        self.__setstats(stat_aggregator=stat_aggregator)

    def __initlogging(self, iter_log_percent):
        """
        Initialize logging.

        Args:
            iter_log_percent (float): An iteration log entry is made before every iter_log_percent chunk of iterations.

        Raises:
            InvalidIterationLogPercentError: The iteration log percent is out of (0.0, 1.0] bounds.
        """

        # Configure logging. This class relies on configured logging somewhere down on the hierarchy.
        #
        qualified_class_name = '{module_name}.{class_name}'.format(module_name=self.__class__.__module__, class_name=self.__class__.__name__)
        self.__logger = logging.getLogger(name=qualified_class_name)

        # Check the iteration log percent: must be in the (0.0, 1.0] interval.
        #
        if iter_log_percent <= 0.0 or 1.0 < iter_log_percent:
            raise dpttrainingerrors.InvalidIterationLogPercentError(iter_log_percent)

        # Set the iteration log percent.
        #
        self.__iter_log_percent = iter_log_percent

    def __flushlogging(self):
        """Flush all log handlers."""

        for log_handler in self.__logger.handlers:
            log_handler.flush()

    def __setmodel(self, model):
        """
        Save the network object to train.

        Args:
            model (dmsmodelbase.ModelBase): Network to train.

        Raises:
            InvalidNetworkObjectError: The network object is invalid.
        """

        # Check if the network object is valid.
        #
        if not model or not isinstance(model, dmsmodelbase.ModelBase):
            raise dpttrainingerrors.InvalidNetworkObjectError()

        # Save the network object.
        #
        self.__model = model

    def __setgenerators(self, training_batch_generator, validation_batch_generator):
        """
        Save the training and validation batch generators.

        Args:
            training_batch_generator (dptbatchgenerator.BatchGenerator): Training batch generator.
            validation_batch_generator (dptbatchgenerator.BatchGenerator): Validation batch generator.

        Raises:
            InvalidBatchGeneratorError: The batch generator object is invalid.
            DimensionOrderMismatchError: Model - generator dimension order mismatch.
        """

        # Check if the batch generators are valid.
        #
        if not training_batch_generator:
            raise dpttrainingerrors.InvalidBatchGeneratorError('training')

        if not validation_batch_generator:
            raise dpttrainingerrors.InvalidBatchGeneratorError('validation')

        # Check if the dimension order of the generators match the dimension order of the network model.
        #
        if self.__model.dimensionorder != training_batch_generator.dimensionorder:
            raise dpttrainingerrors.DimensionOrderMismatchError(self.__model.dimensionorder, training_batch_generator.dimensionorder, training_batch_generator.name)

        if self.__model.dimensionorder != validation_batch_generator.dimensionorder:
            raise dpttrainingerrors.DimensionOrderMismatchError(self.__model.dimensionorder, validation_batch_generator.dimensionorder, validation_batch_generator.name)

        # Save the batch generators.
        #
        self.__training_gen = training_batch_generator
        self.__validation_gen = validation_batch_generator

    def __setmetric(self, metric_name, higher_is_better, averaging_length):
        """
        Set the averaging size. The absolute metric of the individual epochs are averaged over the length of the averaging size to get the actual metric that is
        used to determine the performance of the network.

        Args:
            metric_name (str): Name of the metric to evaluate the performance of the network.
            higher_is_better (bool): Higher metric value is better.
            averaging_length (int): Number of epochs to average over to get the actual metric.

        Raises:
            InvalidAveragingLengthError: The epoch averaging length is not valid.
        """

        # The size must be at least 1.
        #
        if averaging_length < 1:
            raise dpttrainingerrors.InvalidAveragingLengthError(averaging_length)

        # Set the configured values.
        #
        self.__plateau_metric_name = metric_name
        self.__plateau_averaged_metric_name = 'averaged {metric}'.format(metric=metric_name) if 1 < averaging_length else None
        self.__plateau_metric_higher_is_better = higher_is_better
        self.__plateau_epoch_averaging_length = averaging_length

    def __setsyncer(self, file_synchronizer):
        """
        Set the file synchronizer.

        Args:
            file_synchronizer (dptfilesynchronizer.FileSynchronizer): File synchronizer object.

        Raises:
            InvalidFileSynchronizerError: The file synchronizer is not valid.
        """

        # Check if the stats handler is valid.
        #
        if not file_synchronizer:
            raise dpttrainingerrors.InvalidFileSynchronizerError()

        # Save the statistics.
        #
        self.__syncer = file_synchronizer

    def __setstats(self, stat_aggregator):
        """
        Set the statistics aggregator.

        Args:
            stat_aggregator (dptstats.StatAggregator): Statistics aggregator.

        Raises:
            InvalidStatsHandlerError: The statistics handler is invalid.
        """

        # Check if the stats handler is valid.
        #
        if not stat_aggregator:
            raise dpttrainingerrors.InvalidStatAggregatorError()

        # Select the basic statistics to plot.
        #
        stats_to_plot = ['training loss', 'validation loss', 'training accuracy', 'validation accuracy']

        # Add the custom metric name of not already in the list.
        #
        if self.__plateau_metric_name not in stats_to_plot:
            stats_to_plot.append(self.__plateau_metric_name)

        # Add averaged plateau metric to the list of plotted statistics.
        #
        if self.__plateau_averaged_metric_name is not None and self.__plateau_averaged_metric_name not in stats_to_plot:
            stats_to_plot.append(self.__plateau_averaged_metric_name)

        # Save the statistics and configure metrics to plot.
        #
        self.__stats = stat_aggregator
        self.__stats.addplot(stat_names=stats_to_plot)

    def __loadstate(self, model_path, state_path, role):
        """
        Load the last valid state from the disk.

        Args:
            model_path (str): Path of the network model.
            state_path (str): Path of the trainer state.
            role (str): Role of the state for logging: 'best' or 'last'.

        Returns:
            epoch_index (int): Epoch index.
            learning_rate (float): Current learning rate.
            learning_rate_update_index (int): Last learning rate update index.

        Raises:
            ModelHashMismatchError: The stored hash in the state does not match the hash of the actual model file.
        """

        # Load the last state.
        #
        self.__logger.info('Loading {role} network model and state configuration'.format(role=role))
        self.__logger.debug('Loading state configuration file: {path}'.format(path=state_path))

        state_config = dptserialize.load_object(path=state_path)

        self.__logger.debug('Recovered state configuration: {config}'.format(config=state_config))

        # Parse the parameters.
        #
        epoch_index = state_config['epoch_index']
        learning_rate = state_config['learning_rate']
        learning_rate_update_index = state_config['learning_rate_update_index']
        model_digest = state_config['model_digest']

        self.__plateau_metric_last_values = state_config['plateau_metric_last_values']
        self.__plateau_metric_over_epochs = state_config['plateau_metric_over_epochs']

        # The state represents the state of execution at the end of the reported epoch. So the starting epoch index should be the next.
        #
        epoch_index += 1

        # Check de model file against the stored hash.
        #
        self.__logger.debug('Checking network model hash: {path}'.format(path=model_path))
        sha_calc = hashlib.sha256()
        with open(file=model_path, mode='rb') as model_binary:
            sha_calc.update(model_binary.read())

        file_model_digest = sha_calc.hexdigest()

        # Load the network.
        #
        if file_model_digest == model_digest:
            self.__logger.debug('Network model hash matches reference: {hash}'.format(hash=file_model_digest))
            self.__logger.debug('Loading network model file: {path}'.format(path=model_path))
            self.__model.load(file=model_path)
        else:
            # The hash of the available model file does not match the stored hash. (Probably the trainer was interrupted during saving.)
            #
            self.__logger.error('Network model file hash \'{model}\' does not match the state configuration reference: \'{state}\''.format(model=file_model_digest, state=model_digest))
            raise dpttrainingerrors.ModelHashMismatchError(model_digest, file_model_digest)

        self.__logger.debug('Successfully loaded the {role} network model and state'.format(role=role))

        # Return the parsed values.
        #
        return epoch_index, learning_rate, learning_rate_update_index

    def __savestate(self, model_path, state_path, epoch_index, learning_rate, learning_rate_update_index, role):
        """
        Save the current state of the trainer object.

        Args:
            model_path (str): Path of the network model.
            state_path (str): Path of the trainer state.
            epoch_index (int): Epoch index.
            learning_rate (float): Current learning rate.
            learning_rate_update_index (int): Last learning rate update index.
            role (str): Role of the state for logging: 'best' or 'last'.
        """

        # Save the network.
        #
        self.__logger.info('Saving {role} network model and state configuration'.format(role=role))
        self.__logger.debug('Saving network model file: {path}'.format(path=model_path))
        self.__model.save(file_path=model_path)

        # Calculate the hash of the network.
        #
        sha_calc = hashlib.sha256()
        with open(file=model_path, mode='rb') as model_binary:
            sha_calc.update(model_binary.read())
        model_digest = sha_calc.hexdigest()

        self.__logger.debug('Network model hash: {hash}'.format(hash=model_digest))

        # Save the parameters.
        #
        state_config = {'epoch_index': epoch_index,
                        'learning_rate': learning_rate,
                        'learning_rate_update_index': learning_rate_update_index,
                        'model_digest': model_digest,
                        'plateau_metric_last_values': self.__plateau_metric_last_values,
                        'plateau_metric_over_epochs': self.__plateau_metric_over_epochs}

        self.__logger.debug('Compiled state configuration: {{epoch: {epoch}, learning rate: {rate}, update index: {update}, model_digest: \'{digest}\'}}'.format(epoch=epoch_index,
                                                                                                                                                                 rate=learning_rate,
                                                                                                                                                                 update=learning_rate_update_index,
                                                                                                                                                                 digest=model_digest))

        # Serialize the data. It contains floating point numbers that would be distorted if saved to text files like YAML.
        #
        self.__logger.debug('Saving state configuration file: {path}'.format(path=state_path))
        dptserialize.save_object(content=state_config, path=state_path)

        self.__logger.debug('Successfully saved the {role} network model and training state'.format(role=role))

    def __syncstats(self, progress_row):
        """
        Save epoch statistics and synchronize work files to target paths.

        Args:
            progress_row (dict): Progress row.
        """

        # Add new entry to the statistics table.
        #
        self.__logger.info('Saving epoch statistics and synchronizing files')
        self.__logger.debug('Adding epoch statistics')
        self.__stats.append(epoch_statistics_row=progress_row)

        # Save statistics and plot progress.
        #
        self.__logger.debug('Saving epoch statistics to table: {path}'.format(path=self.__stats.savepath))
        self.__stats.save()

        self.__logger.debug('Plotting progress to image: {path}'.format(path=self.__stats.plotpath))
        self.__stats.plot()

        # Sync output files to target path.
        #
        self.__logger.debug('Synchronizing files from work directory to target paths')
        self.__syncer.sync(target_path=None, move=False)

    def __inittraining(self, learning_rate, boosting_enabled, last_model_path, last_state_path,  best_model_path, best_state_path, continue_experiment):
        """
        Initialize training by loading the last parameters and network or initializing the parameters and network with the default values.

        Args:
            learning_rate (float): Learning rate.
            boosting_enabled (bool): Flag to control if boosting is enabled.
            last_model_path (str): Path of the last network model.
            last_state_path (str): Path of the last execution state.
            best_model_path (str): Path of the best network model.
            best_state_path: Path of the best execution state.
            continue_experiment (bool): Whether to continue with a previously trained network.

        Returns:
            epoch_index (int): Epoch index.
            learning_rate (float): Current learning rate.
            learning_rate_update_index (int): Last learning rate update index.

        Raises:
            ModelHashMismatchError: The stored hash in the state does not match the hash of the actual model file.
            UnknownMetricNameError: The required metric is not produced by the model.
        """

        # To avoid warnings.
        #
        self.__logger.info("Initializing parameters...")

        starting_epoch = 0
        current_learning_rate = 0.0
        learning_rate_update_index = 0

        # If boosting is enabled, add difficult ratio to the list of plotted stats.
        #
        if boosting_enabled:
            stat_name = 'training difficult patch ratio'

            self.__logger.debug('Boosting enabled, adding \'{name}\' to the list of stats to plot'.format(name=stat_name))
            self.__stats.addplot(stat_names=stat_name)

        # Try to load the state of the last executed epoch or if it is inconsistent then the best epoch.
        #
        successful_loading = False
        if continue_experiment:
            # Synchronize back the output table and load the statistics.
            #
            self.__logger.info('Synchronizing intermediate results')
            self.__syncer.back(target_path=None, move=False)
            self.__stats.load()

            # Check if the last state is available for loading.
            #
            if os.path.isfile(last_model_path) and os.path.isfile(last_state_path):
                try:
                    starting_epoch, current_learning_rate, learning_rate_update_index = self.__loadstate(model_path=last_model_path, state_path=last_state_path, role='last')
                except Exception as exception:
                    self.__logger.error('Last training state is invalid: {exception}'.format(exception=exception))
                else:
                    # State successfully loaded. Prevent further loadings or initialization by setting the flag.
                    #
                    successful_loading = True
            else:
                self.__logger.info('Last training state is not available')

            # If the loading was unsuccessful.
            #
            if not successful_loading:
                # Check if the best state is available for loading.
                #
                if os.path.isfile(best_model_path) and os.path.isfile(best_state_path):
                    try:
                        starting_epoch, current_learning_rate, learning_rate_update_index = self.__loadstate(model_path=best_model_path, state_path=best_state_path, role='best')
                    except Exception as exception:
                        self.__logger.error('Best training state is invalid: {exception}'.format(exception=exception))
                    else:
                        # State successfully loaded. Prevent further loadings or initialization by setting the flag.
                        #
                        successful_loading = True
                else:
                    self.__logger.info('Best training state is not available')

        if successful_loading:
            # If the continuation is enabled and the loading was successful rewind the stats to the target epoch.
            #
            self.__logger.info('Rewinding stats to epoch: {index}'.format(index=starting_epoch))

            self.__stats.rewind(index=starting_epoch)

        else:
            # Initialize the model from the definition and set the default value to the stating epoch index and learning rate.
            #
            self.__logger.info("Setting parameters to default")

            starting_epoch = 0
            current_learning_rate = learning_rate
            learning_rate_update_index = 0

            self.__logger.info("Building network...")
            self.__model.build()
            self.__model.updatelearningrate(learning_rate=current_learning_rate)

        # Collect the available metrics and check if the required metric name is available.
        #
        available_metrics = []
        for role in ['training', 'validation']:
            available_metrics.extend([' '.join((role, metric)) for metric in self.__model.metricnames()])

        if self.__plateau_metric_name not in available_metrics:
            raise dpttrainingerrors.UnknownMetricNameError(self.__plateau_metric_name, available_metrics)

        # If boosting is enabled check if the network returns the errors.
        #
        if boosting_enabled and 'training errors' not in available_metrics:
            raise dpttrainingerrors.UnknownMetricNameError('training errors', available_metrics)

        # The model is initialized, return the initialized or loaded parameters of the training.
        #
        return starting_epoch, current_learning_rate, learning_rate_update_index

    def __startfillingbuffer(self, batch_generator):
        """
        Start loading patches to the buffer for the next epoch. This function takes care of the different multi-threaded/single-threaded and
        double-buffered/single-buffered configurations. Some of them does not make too much sense but still possible.

        With double buffering the network trainer always loads the patches for the next epoch in the background. If there is no double buffering
        the loaded patches are consumed in the current epoch.

        Args:
            batch_generator (dptbatchgenerator.BatchGenerator): Batch generator.
        """

        # Start loading the new patches in the background. If it is not multi-threaded then it will by a blocking call.
        #
        self.__logger.info('Filling the {role} batch generator with {count} patches'.format(role=batch_generator.name, count=batch_generator.size))
        if not batch_generator.multithreaded:
            self.__logger.info('Waiting for the {role} batch generator...'.format(role=batch_generator.name))

        batch_generator.load(batch_size=0)

        # If the batch generator is not double-buffered than all the read patches are going to the main buffer that should interfere with the batch generation,
        # therefore we must wait for it to be finished here.
        #
        if not batch_generator.doublebuffered:
            batch_generator.wait()

    def __transferbuffer(self, batch_generator, count, difficult_threshold):
        """
        Transfer the loaded patches from the secondary (read) buffer to the main buffer of the batch generator.

        Args:
            batch_generator (dptbatchgenerator.BatchGenerator): Batch generator.
            count (int): Number of patches to transfer.
            difficult_threshold (float): Error threshold for difficult patches.
        """

        # Transfer the loaded patches to the main buffer.
        #
        if batch_generator.multithreaded:
            self.__logger.info('Waiting for the {role} batch generator...'.format(role=batch_generator.name))

        # If the batch generator is single-buffered than the wait is executed immediately since the load command had a wait barrier earlier and the transfer
        # command does noting.
        #
        batch_generator.wait()
        batch_generator.transfer(batch_size=count, difficult_threshold=difficult_threshold)

    def __checkgenerator(self, batch_generator):
        """
        Check the batch generator and reset its timers.

        Args:
            batch_generator (dptbatchgenerator.BatchGenerator): Batch generator.

        Raises:
            DigitalPathologyProcessError: Process errors.
        """

        self.__logger.info('Checking the {role} batch generator'.format(role=batch_generator.name))

        batch_generator.ping()

    def __unpackdata(self, batch):
        """
        Prepare data so that it matches the format required by the models.

        Args:
            batch (dict): Labels and data.

        Returns:
            patches (list): Patch data.
            labels (list): Labels.
            weights (list): Weights.
        """

        # Ravel the dictionary.
        #
        patches = [level_data['patches'] for level_data in batch.values()]
        labels = [level_data['labels'] for level_data in batch.values()]
        weights = [level_data.get('weights', None) for level_data in batch.values()]

        if len(patches) == 1:
            patches = patches[0]
        if len(labels) == 1:
            labels = labels[0]
        if len(weights) == 1:
            weights = weights[0]

        return patches, labels, weights

    @staticmethod
    def __appendepochstats(epoch_stats, model_output, role):
        """
        Remove the items from the model output that should not be logged, like 'errors' or 'predictions' and add the statistics from the model output the the epoch stat lists.

        Args:
            epoch_stats (dict): Epoch statistics so far.
            model_output (dict): Model output.
            role (str): Iteration role to use as a prefix before the values.

        Return:
            dict: Epoch stat lists with values added from the model output.
        """

        # Add prefix and remove 'errors' and 'predictions' from stats and add them to the epoch stats.
        #
        for stat_name in model_output:
            if stat_name != 'errors' and stat_name != 'predictions':
                actual_stat_name = '{role} {stat}'.format(role=role, stat=stat_name)

                if actual_stat_name not in epoch_stats:
                    epoch_stats[actual_stat_name] = []
                epoch_stats[actual_stat_name].append(model_output[stat_name])

        return epoch_stats

    def __trainingepoch(self, epoch_index, repetition_count, iter_count, batch_size, boosting_enabled, mode_switch_epoch, difficult_threshold, difficult_update_ratio):
        """
        Execute a training epoch.

        Args:
            epoch_index (int): Current epoch index.
            repetition_count (int): Repetition count.
            iter_count (int): Iterations to execute.
            batch_size (int): Batch size to use.
            boosting_enabled (bool): Flag to control if boosting is enabled.
            mode_switch_epoch (int): Epoch index where the buffer mode should be switched from the initial 'ring' mode to the configured one.
            difficult_threshold (float): Error threshold for difficult patches.
            difficult_update_ratio (float): Ratio of the identified patches to update.

        Returns:
            (dict): Returns a dictionary with the collected statistics of the epoch (e.g execution time and loss)

        Raises:
            ErrorsNotInModelOutputError: The network does not return errors despite it is required.

            DigitalPathologyBufferError: Buffer errors.
            DigitalPathologyConfigError: Configuration errors.
        """

        # Measure epoch execution time.
        #
        epoch_start_time = time.time()
        epoch_gpu_time = 0.0

        # Accumulate epoch statistics.
        #
        epoch_stats = {}

        # Repeat the load-train-wait-transfer blocks.
        #
        for repetition_index in range(repetition_count):
            self.__logger.info('Training repetition {index} of {total} --------------------------------'.format(index=repetition_index + 1, total=repetition_count))

            # Start loading the new patches in the background: if double-buffering is enabled the patches are loaded into the secondary buffer and will be
            # consumed in the next epoch. With only the main buffer (single-buffering) the patches are directly loaded to the main buffer and will be consumed
            # in this epoch.
            #
            self.__startfillingbuffer(batch_generator=self.__training_gen)

            # Check the validation batch generator to keep it alive. Too long training epoch might cause the validation batch generator to time out.
            #
            self.__checkgenerator(batch_generator=self.__validation_gen)

            # Calculate the iterations to report in log.
            #
            iter_log_interval = max(1.0, iter_count * self.__iter_log_percent)
            next_iter_to_report = 0.0

            # Go through the iterations.
            #
            model_output = []
            for iter_index in range(iter_count):
                # Print status.
                #
                if iter_index == round(next_iter_to_report):
                    next_iter_to_report += iter_log_interval
                    self.__logger.info('Training iteration {index} - {last} of {total}'.format(index=iter_index + 1, last=min(iter_count, round(next_iter_to_report)), total=iter_count))

                # Load patches and calculate loss with prediction.
                #
                batch, indices = self.__training_gen.batch(batch_size=batch_size)

                # Make data compatible with model base.
                #
                patches, labels, weights = self.__unpackdata(batch=batch)

                gpu_time_start = time.time()
                model_output = self.__model.update(x=patches, y=labels, sample_weight=weights)

                # Measure GPU time for efficiency monitoring.
                #
                epoch_gpu_time += (time.time() - gpu_time_start)

                # Store the classification errors.
                #
                if boosting_enabled:
                    if 'errors' in model_output:
                        self.__training_gen.update(indices=indices, errors=model_output['errors'])
                    else:
                        raise dpttrainingerrors.ErrorsNotInModelOutputError()

                # Collect the statistics.
                #
                epoch_stats = self.__appendepochstats(epoch_stats=epoch_stats, model_output=model_output, role='training')

            # Count the number of difficult patches remaining in the buffer.
            #
            transfer_count = self.__training_gen.size
            if boosting_enabled and difficult_update_ratio < 1.0 and difficult_threshold < 1.0:
                # Errors are necessary for boosting calculation.
                #
                if 'errors' in model_output:

                    difficult_patch_count = self.__training_gen.count(threshold=difficult_threshold)

                    if mode_switch_epoch <= epoch_index:
                        difficult_replace_count = round(difficult_patch_count * difficult_update_ratio)
                        transfer_count = self.__training_gen.size - (difficult_patch_count - difficult_replace_count)

                        self.__logger.info('Boosting: replacing only {replace} out of {total} difficult training patches'.format(replace=difficult_replace_count, total=difficult_patch_count))

                    difficult_patch_ratio = difficult_patch_count / self.__training_gen.size
                else:
                    # Errors not available mark it as NaN in the statistics.
                    #
                    difficult_patch_ratio = math.nan

                # Add the difficult ratio to the statistics.
                #
                training_stat_name = 'training difficult patch ratio'
                if training_stat_name not in epoch_stats:
                    epoch_stats[training_stat_name] = []
                epoch_stats[training_stat_name].append(difficult_patch_ratio)

            # Transfer the loaded patches to the main buffers.
            #
            self.__transferbuffer(batch_generator=self.__training_gen, count=transfer_count, difficult_threshold=difficult_threshold)

        # Average the accumulated statistics over the repetitions.
        #
        for stat_name in epoch_stats:
            epoch_stats[stat_name] = np.mean(epoch_stats[stat_name])

        # End of epoch execution.
        #
        epoch_stats['training epoch gpu time'] = epoch_gpu_time
        epoch_stats['training epoch execution time'] = time.time() - epoch_start_time

        # Return the results.
        #
        return epoch_stats

    def __validationepoch(self, repetition_count, iter_count, batch_size):
        """
        Execute a validation epoch.

        Args:
            repetition_count (int): Repetition count.
            iter_count (int): Iterations to execute.
            batch_size (int): Batch size to use.

        Returns:
            (dict): Returns a dictionary with the collected statistics of the epoch (e.g execution time and loss)

        Raises:
            DigitalPathologyBufferError: Buffer errors.
            DigitalPathologyConfigError: Configuration errors.
        """

        # Measure epoch execution time.
        #
        epoch_start_time = time.time()
        epoch_gpu_time = 0.0

        # Accumulate epoch statistics.
        #
        epoch_stats = {}

        # Repeat the load-validate-wait-transfer blocks.
        #
        for repetition_index in range(repetition_count):
            self.__logger.info('Validation repetition {index} of {total} --------------------------------'.format(index=repetition_index + 1, total=repetition_count))

            # Start loading the new patches in the background: if double-buffering is enabled the patches are loaded into the secondary buffer and will be
            # consumed in the next epoch. With only the main buffer (single-buffering) the patches are directly loaded to the main buffer and will be consumed
            # in this epoch.
            #
            self.__startfillingbuffer(batch_generator=self.__validation_gen)

            # Check the training batch generator to keep it alive. Too long validation epoch might cause the training batch generator to time out.
            #
            self.__checkgenerator(batch_generator=self.__training_gen)

            # Calculate the iterations to report in log.
            #
            iter_log_interval = max(1.0, iter_count * self.__iter_log_percent)
            next_iter_to_report = 0.0

            # Go through the iterations.
            #
            for iter_index in range(iter_count):
                # Print status.
                #
                if iter_index == round(next_iter_to_report):
                    next_iter_to_report += iter_log_interval
                    self.__logger.info('Validation iteration {index} - {last} of {total}'.format(index=iter_index + 1, last=min(iter_count, round(next_iter_to_report)), total=iter_count))

                batch, _ = self.__validation_gen.batch(batch_size=batch_size)

                # Make data compatible with model base.
                #
                patches, labels, weights = self.__unpackdata(batch=batch)

                gpu_time_start = time.time()
                model_output = self.__model.validate(x=patches, y=labels, sample_weight=weights)

                # Measure GPU time for efficiency monitoring.
                #
                epoch_gpu_time += (time.time() - gpu_time_start)

                # Collect the statistics.
                #
                epoch_stats = self.__appendepochstats(epoch_stats=epoch_stats, model_output=model_output, role='validation')

            # Transfer the loaded patches to the main buffers.
            #
            self.__transferbuffer(batch_generator=self.__validation_gen, count=0, difficult_threshold=0.0)

        # Average the accumulated statistics over the repetitions.
        #
        for stat_name in epoch_stats:
            epoch_stats[stat_name] = np.mean(epoch_stats[stat_name])

        # End of epoch execution.
        #
        epoch_stats['validation epoch gpu time'] = epoch_gpu_time
        epoch_stats['validation epoch execution time'] = time.time() - epoch_start_time

        # Return the results.
        #
        return epoch_stats

    def __plateaulength(self, current_plateau_metric_value):
        """
        Store the current plateau metric, apply filter on the stored array and calculate the plateau length.

        Args:
            current_plateau_metric_value (float): Plateau metric of the current epoch.

        Returns:
            (float): Average plateau metric size.
            (int): Plateau length.
            (float): Best metric value.
            (int): Epoch of best metric.
        """

        # Add the metric to the last value lists.
        #
        self.__plateau_metric_last_values.append(current_plateau_metric_value)
        self.__plateau_metric_last_values = self.__plateau_metric_last_values[-self.__plateau_epoch_averaging_length:]
        self.__plateau_metric_over_epochs.append(np.mean(self.__plateau_metric_last_values))

        # Calculate the accuracy plateau length.
        #
        previous_best_index = len(self.__plateau_metric_over_epochs) - 1
        plateau_length = 0

        if self.__plateau_metric_higher_is_better:
            previous_max_value = np.max(self.__plateau_metric_over_epochs[0:-1]) if len(self.__plateau_metric_over_epochs) > 1 else -np.inf
            if previous_max_value > self.__plateau_metric_over_epochs[-1]:
                previous_best_index = np.argmax(self.__plateau_metric_over_epochs[0:-1])
                plateau_length = len(self.__plateau_metric_over_epochs) - 1 - previous_best_index
        else:
            previous_min_value = np.min(self.__plateau_metric_over_epochs[0:-1]) if len(self.__plateau_metric_over_epochs) > 1 else np.inf
            if previous_min_value < self.__plateau_metric_over_epochs[-1]:
                previous_best_index = np.argmin(self.__plateau_metric_over_epochs[0:-1])
                plateau_length = len(self.__plateau_metric_over_epochs) - 1 - previous_best_index

        return self.__plateau_metric_over_epochs[-1], plateau_length, self.__plateau_metric_over_epochs[previous_best_index], previous_best_index

    def __updatelearningrate(self, learning_rate, epoch_index, plateau_length, update_index, decay_enabled, update_factor, update_plateau):
        """
        Update the learning rate.

        Args:
            learning_rate (float): Current learning rate.
            epoch_index (int): Current epoch index.
            plateau_length (int): Non improving plateau length.
            update_index (int): Epoch index of the last learning rate update.
            decay_enabled (bool): Flag to control if learning rate is enabled.
            update_factor (float): Learning rate decay.
            update_plateau (int): Number of epochs with non improving results before decaying the learning rate.

        Returns:
            (float, float, int, int): Previous learning rate, updated learning rate, learning rate update index, current plateau length.
        """

        # Calculate the learning rate update criteria is met.
        #
        plateau_for_decay_count = min(epoch_index - update_index, plateau_length)

        # Check if the learning rate decay needs to be applied.
        #
        previous_learning_rate = learning_rate
        if decay_enabled and update_plateau <= plateau_for_decay_count:
            # Update the learning rate for the next epoch = apply decay.
            #
            new_learning_rate = learning_rate * update_factor
            new_update_index = epoch_index

            self.__logger.debug('Learning rate update: {old} -> {new}'.format(old=learning_rate, new=new_learning_rate))

            self.__model.updatelearningrate(learning_rate=new_learning_rate)

        else:
            # Learning rate decay is not necessary.
            #
            new_learning_rate = learning_rate
            new_update_index = update_index

        return previous_learning_rate, new_learning_rate, new_update_index, plateau_for_decay_count

    def execute(self,
                epoch_count,
                source_step_length,
                training_repetition_count,
                validation_repetition_count,
                training_iter_count,
                validation_iter_count,
                training_batch_size,
                validation_batch_size,
                boosting_enabled,
                buffer_mode_switch,
                difficult_threshold,
                difficult_update_ratio,
                learning_rate,
                learning_rate_decay_enabled,
                learning_rate_update_factor,
                learning_rate_update_plateau,
                stop_plateau_enabled,
                stop_plateau_length,
                best_model_path,
                best_state_path,
                last_model_path,
                last_state_path,
                continue_experiment):
        """
        Execute training epochs.

        Args:
            epoch_count (int): Upper limit of epochs to execute.
            source_step_length (int): The batch generator stepped after the given number of epochs. Set 0 to disable it.
            training_repetition_count (int): Repetition count for training iterations in an epoch.
            validation_repetition_count (int): Repetition count for validation iterations in an epoch.
            training_iter_count (int): Number of training iterations in a single epoch.
            validation_iter_count (int): Number of validation iterations in a single epoch.
            training_batch_size (int): Number of patches in a training minibatch.
            validation_batch_size (int): Number of patches in a validation minibatch.
            boosting_enabled (bool): Flag to control if boosting is enabled.
            buffer_mode_switch (int): Epoch index where the buffer mode should be switched from the initial 'ring' mode to the configured one.
            difficult_threshold (float): Error threshold for difficult patches.
            difficult_update_ratio (float): Ratio of the identified patches to update.
            learning_rate (float): Learning rate.
            learning_rate_decay_enabled (bool): Flag to control if learning rate is enabled.
            learning_rate_update_factor (float): Learning rate decay.
            learning_rate_update_plateau (int): Number of epochs with non improving results before decaying the learning rate.
            stop_plateau_enabled (bool): Stop on plateau enabled.
            stop_plateau_length (int): Number of epochs with non improving results before stopping the training.
            best_model_path (str): Path of the best network model.
            best_state_path (str): Path of the best execution state.
            last_model_path (str): Path of the last network model.
            last_state_path (str): Path of the last execution state.
            continue_experiment (bool): Whether to continue with a previously trained network.

        Raises:
            InvalidEpochCountError: The epoch count in less than 1.
            InvalidRepetitionCountError: The repetition count is less than 1.
            InvalidIterationCountError: The training or validation iteration count is less than 1.
            InvalidBufferConfigurationError: Boosting is enabled without double buffering.
            InvalidDifficultThreshold: The difficult example threshold is out of the [0.0, 1.0] interval.
            InvalidDifficultUpdateRatio: The difficult example update ratio is out of the [0.0, 1.0] interval.
            InvalidLearningRateError: Non positive learning rate.
            InvalidModelSavePathError: Invalid network model dump path.
            InvalidStateSavePathError: Invalid training state dump path.
            UnknownMetricNameError: The required metric is not produced by the model.

            DigitalPathologyBufferError: Buffer errors.
            DigitalPathologyConfigError: Configuration errors.
            DigitalPathologyDataError: Data errors.
            DigitalPathologyImageError: Image errors.
            DigitalPathologyLabelError: Label errors. 
            DigitalPathologyModelError: Model errors.
            DigitalPathologyProcessError: Process errors.
            DigitalPathologyStatError: Stats errors.
        """

        # Check parameters.
        #
        if epoch_count <= 0:
            raise dpttrainingerrors.InvalidEpochCountError(epoch_count)

        if training_repetition_count <= 0:
            raise dpttrainingerrors.InvalidRepetitionCountError('training', training_repetition_count)

        if validation_repetition_count <= 0:
            raise dpttrainingerrors.InvalidRepetitionCountError('validation', validation_repetition_count)

        if training_iter_count <= 0:
            raise dpttrainingerrors.InvalidIterationCountError('training', training_iter_count)

        if validation_iter_count <= 0:
            raise dpttrainingerrors.InvalidIterationCountError('validation', validation_iter_count)

        if 0 < buffer_mode_switch < epoch_count and not self.__training_gen.doublebuffered:
            raise dpttrainingerrors.InvalidBufferConfigurationError(buffer_mode_switch)

        if difficult_threshold < 0.0 or 1.0 < difficult_threshold:
            raise dpttrainingerrors.InvalidDifficultThreshold(difficult_threshold)

        if difficult_update_ratio < 0.0 or 1.0 < difficult_update_ratio:
            raise dpttrainingerrors.InvalidDifficultUpdateRatio(difficult_update_ratio)

        if learning_rate <= 0.0:
            raise dpttrainingerrors.InvalidLearningRateError(learning_rate)

        if not best_model_path:
            raise dpttrainingerrors.InvalidModelSavePathError('best')

        if not best_state_path:
            raise dpttrainingerrors.InvalidStateSavePathError('best')

        if not last_model_path:
            raise dpttrainingerrors.InvalidModelSavePathError('last')

        if not last_state_path:
            raise dpttrainingerrors.InvalidStateSavePathError('last')

        try:
            # Initialize training: load the last parameters and network status or initialize it from scratch.
            #
            starting_epoch, current_learning_rate, learning_rate_update_index = self.__inittraining(learning_rate=learning_rate,
                                                                                                    boosting_enabled=boosting_enabled,
                                                                                                    last_model_path=last_model_path,
                                                                                                    last_state_path=last_state_path,
                                                                                                    best_model_path=best_model_path,
                                                                                                    best_state_path=best_state_path,
                                                                                                    continue_experiment=continue_experiment)

            # Initialize the generators.
            #
            self.__logger.info("Initializing generators...")
            self.__training_gen.start()
            self.__training_gen.step()
            self.__training_gen.wait()

            self.__validation_gen.start()
            self.__validation_gen.step()
            self.__validation_gen.wait()

            # Loading initial data batch generators.
            #
            self.__logger.info("Filling patch buffers...")
            self.__training_gen.fill()
            self.__training_gen.wait()

            self.__validation_gen.fill()
            self.__validation_gen.wait()

            # Execute the epochs with training and validation iterations.
            #
            self.__logger.info("Training...")

            best_plateau_metric_value = 0.0
            best_plateau_metric_epoch = 0

            for epoch_index in range(starting_epoch, epoch_count):
                # Measure total epoch execution time.
                #
                epoch_start_time = time.time()
                self.__logger.info('Epoch {index} of {total} ----------------------------------------------------------------'.format(index=epoch_index + 1, total=epoch_count))

                # Execute training iterations.
                #
                training_epoch_stats = self.__trainingepoch(epoch_index=epoch_index,
                                                            repetition_count=training_repetition_count,
                                                            iter_count=training_iter_count,
                                                            batch_size=training_batch_size,
                                                            boosting_enabled=boosting_enabled,
                                                            mode_switch_epoch=buffer_mode_switch,
                                                            difficult_threshold=difficult_threshold,
                                                            difficult_update_ratio=difficult_update_ratio)

                # Save training epoch data.
                #
                progress_row = {'learning rate': current_learning_rate}
                progress_row.update(training_epoch_stats)

                # Execute validation iterations.
                #
                validation_epoch_stats = self.__validationepoch(repetition_count=validation_repetition_count, iter_count=validation_iter_count, batch_size=validation_batch_size)

                # Save validation epoch data.
                #
                progress_row.update(validation_epoch_stats)

                # Check the accuracy. Calculate learning rate decay and stop criteria plateau length.
                #
                self.__logger.info('Checking results...')
                (plateau_metric_value,
                 plateau_length,
                 best_plateau_metric_value,
                 best_plateau_metric_epoch) = self.__plateaulength(current_plateau_metric_value=progress_row[self.__plateau_metric_name])

                # Save the actual metric value if it is derived from the model outputs.
                #
                if self.__plateau_averaged_metric_name:
                    progress_row[self.__plateau_averaged_metric_name] = plateau_metric_value

                # Update the learning rate for the next epoch = apply decay.
                #
                epoch_learning_rate, current_learning_rate, learning_rate_update_index, plateau_for_decay_count = self.__updatelearningrate(learning_rate=current_learning_rate,
                                                                                                                                            epoch_index=epoch_index,
                                                                                                                                            update_index=learning_rate_update_index,
                                                                                                                                            plateau_length=plateau_length,
                                                                                                                                            decay_enabled=learning_rate_decay_enabled,
                                                                                                                                            update_factor=learning_rate_update_factor,
                                                                                                                                            update_plateau=learning_rate_update_plateau)

                # Step the generators.
                #
                stepped_source = False
                if 0 < source_step_length and (epoch_index + 1) % source_step_length == 0:
                    self.__logger.info('Stepping batch sources...')

                    self.__training_gen.step()
                    self.__training_gen.wait()

                    self.__validation_gen.step()
                    self.__validation_gen.wait()

                    stepped_source = True

                # Measure the epoch execution time.
                #
                epoch_end_time = time.time()
                epoch_exec_time = epoch_end_time - epoch_start_time
                gpu_time_ratio = (progress_row['training epoch gpu time'] + progress_row['validation epoch gpu time']) / epoch_exec_time

                # Save progress, histogram.
                #
                self.__logger.info('Saving data...')

                # Save the network if improved.
                #
                network_improved = plateau_length == 0

                if network_improved:
                    self.__savestate(model_path=best_model_path,
                                     state_path=best_state_path,
                                     epoch_index=epoch_index,
                                     learning_rate=current_learning_rate,
                                     learning_rate_update_index=learning_rate_update_index,
                                     role='best')

                # Save network so experiment can be restarted.
                #
                self.__savestate(model_path=last_model_path,
                                 state_path=last_state_path,
                                 epoch_index=epoch_index,
                                 learning_rate=current_learning_rate,
                                 learning_rate_update_index=learning_rate_update_index,
                                 role='last')

                # Push the statistics to the statistics aggregator object. Minor inaccuracy is that the saving and plotting is not included in the epoch execution time.
                #
                progress_row['total epoch execution time'] = epoch_exec_time

                self.__syncstats(progress_row=progress_row)

                # Print statistics.
                #
                self.__logger.info('Epoch execution time: {delta:.2f} sec'.format(delta=epoch_exec_time))
                self.__logger.info('GPU efficiency: {ratio}'.format(ratio=gpu_time_ratio))
                self.__logger.info('Learning rate: {rate}'.format(rate=epoch_learning_rate))
                self.__logger.info('Stopping criteria plateau: {count}/{last}'.format(count=plateau_length, last=stop_plateau_length if stop_plateau_enabled else None))
                self.__logger.info('Learning rate update plateau: {count}/{last}'.format(count=plateau_for_decay_count, last=learning_rate_update_plateau if learning_rate_decay_enabled else None))
                self.__logger.info('Training loss: {loss}'.format(loss=progress_row['training loss']))
                self.__logger.info('Validation loss: {loss}'.format(loss=progress_row['validation loss']))
                self.__logger.info('Last {metric}: {value}'.format(metric=self.__plateau_metric_name, value=progress_row[self.__plateau_metric_name]))

                if self.__plateau_averaged_metric_name:
                    self.__logger.info('Average of last {count} {metric}: {value}'.format(metric=self.__plateau_metric_name,
                                                                                          count=self.__plateau_epoch_averaging_length,
                                                                                          value=progress_row[self.__plateau_averaged_metric_name]))

                self.__logger.info('Network improved: {test}'.format(test=network_improved))
                self.__logger.info('Sources stepped: {step_made}'.format(step_made=stepped_source))

                # Flush the loggers for progress examination after each epoch.
                #
                self.__flushlogging()

                # Log the administration time that is not reported as part if the epoch execution time.
                #
                administration_time = time.time() - epoch_end_time

                self.__logger.info('Administration time: {delta:.2f} sec'.format(delta=administration_time))

                # Check if stop criteria met.
                #
                if stop_plateau_enabled and stop_plateau_length <= plateau_length:
                    break

            # Report best validation accuracy and index of the epoch.
            #
            self.__logger.info('Best {metric} was {value} at epoch {index}'.format(metric=self.__plateau_metric_name, value=best_plateau_metric_value, index=best_plateau_metric_epoch + 1))

        except Exception as exception:
            # Collect and summarize traceback information.
            #
            _, _, exception_traceback = sys.exc_info()
            trace_string = dpttrace.format_traceback(traceback_object=exception_traceback)

            # Log the exception.
            #
            self.__logger.info('Exception raised: "{ex}"'.format(ex=exception))
            self.__logger.error('Exception: "{ex}"; trace: "{trace}"'.format(ex=exception, trace=trace_string))

        finally:
            self.__logger.info('Shutting down batch generators...')

            # Terminate the batch generator objects to let the program shut down in a clean way.
            #
            if self.__training_gen:
                self.__training_gen.stop()

            self.__logger.info('Training batch generator stopped')

            if self.__validation_gen:
                self.__validation_gen.stop()

            self.__logger.info('Validation batch generator stopped')

        self.__logger.info('Training finished')
