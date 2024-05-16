"""
This file contains class for executing experiments.
"""

from . import networktrainer as dptnetworktrainer

from ..batch import batchgenerator as dptbatchgenerator
from ..batch import batchsource as dptbatchsource
from ..augmenters import augmenterpool as dptaugmenterpool
from ..augmenters import passthroughaugmenter as dptpassthroughaugmenter
from ..augmenters.color import contrastaugmenter as dptcontrastaugmenter
from ..augmenters.color import hedcoloraugmenter as dpthedcoloraugmenter
from ..augmenters.color import hsbcoloraugmenter as dpthsbcoloraugmenter
from ..augmenters.noise import additiveguassiannoiseaugmenter as dtpadditiveguassiannoiseaugmenter
from ..augmenters.noise import gaussianbluraugmenter as dptgaussianbluraugmenter
from ..augmenters.spatial import elasticagumenter as dptelasticagumenter
from ..augmenters.spatial import flipaugmenter as dptflipaugmenter
from ..augmenters.spatial import rotate90augmenter as dptrotate90augmenter
from ..augmenters.spatial import scalingaugmenter as dptscalingaugmenter
from ..normalizers import normalizerbase as dptnormalizerbase
from ..normalizers import generalnormalizer as dtpgeneralnormalizer
from ..normalizers import rgbnormalizer as dtprgbnormalizer
from ..normalizers import rgbtozeroonenormalizer as dtprgbtozeroonenormalizer
from ..normalizers import passthroughnormalizer as dtppassthroughnormalizer
from ..label import labelmapper as dptlabelmapper
from ..weight import weightmapperbase as dptweightmapperbase
from ..weight import cleanweightmapper as dptcleanweightmapper
from ..weight.normalizing import batchweightmapper as dptbatchweightmapper
from ..weight.normalizing import patchweightmapper as dptpatchweightmapper
from ..stats import stataggregator as dptstats
from ..utils import population as dptpopulation
from ..utils import gitrepo as dptgitrepo
from ..utils import trace as dpttrace
from ..utils import imagefile as dptimagefile
from ..utils import filesynchronizer as dptfilesynhcronizer

import numpy as np
import logging
import zipfile
import datetime
import yaml
import os
import shutil
import random
import sys
import time

#----------------------------------------------------------------------------------------------------

class ExperimentCommander(object):
    """This class can execute an experiment."""

    def __init__(self,
                 experiment_name,
                 progress_dir_path,
                 work_dir_path,
                 archive_dir_path,
                 data_file_path,
                 data_overrides,
                 data_copy_config,
                 param_file_path,
                 param_overrides,
                 create_stats,
                 continue_experiment,
                 file_log_level='info',
                 random_seed=None,
                 cpu_count_enforce=None,
                 caller_dir_paths=None):
        """
        Configure experiment.

        Args:
            experiment_name (str): Experiment name.
            progress_dir_path (str): Progress data dump directory.
            work_dir_path (str, None): Work directory path.
            archive_dir_path (str, None): Zip all outputs here after the training is done.
            data_file_path (str): Path of the data source file to load.
            data_overrides (dict, None): Path overrides in data source.
            data_copy_config (dict, None): Data copy configuration: replacement for copy source.
            param_file_path (str): Path of the parameters file to load.
            param_overrides (dict, None): Value overrides in parameter configuration.
            create_stats (bool): Allow missing stat files, and create them if necessary.
            continue_experiment (bool): Whether to continue with a previously trained network.
            file_log_level (str): File log level.
            random_seed (int, None): Random seed for reproducible experiments.
            cpu_count_enforce (int, None): Number of CPUs to assume to be available for the experiment. If None the number is queried from the system.
            caller_dir_paths (list): List of source directory paths of the caller library for repository information.

        Raises:
            ValueError: The logging location is invalid.
            ValueError: The file logging level is invalid.
            ValueError: The console logging level is invalid.
        """

        # Initialize the base class.
        #
        super().__init__()

        # Initialize members.
        #
        self.__experiment_name = ''     # Experiment name parameter.
        self.__progress_dir_path = ''   # Progress directory path parameter.
        self.__archive_dir_path = None  # Archive directory path parameter.
        self.__work_dir_path = None     # Work directory path parameter.
        self.__caller_dir_paths = []    # Caller directory paths parameter.

        self.__experiment_id = ''           # Experiment identifier.
        self.__data_file_path = ''          # Data file path.
        self.__data_overrides = None        # Data path overrides dictionary.
        self.__data_copy_config = None      # Data copy configuration.
        self.__param_file_path = ''         # Experiment parameters path.
        self.__param_overrides = None       # Experiment parameter value overrides.
        self.__create_stats = False         # Create missing .stat files.
        self.__continue_experiment = False  # Continue experiment if possible.
        self.__random_seed = None           # Random seed.
        self.__cpu_count_enforce = None     # Enforced CPU count.
        self.__logger = None                # Configured logger object.
        self.__dict_overwrites = None       # Set of parameter keys to overwrite instead of merge with the overrides.
        self.__copy_replacements = None     # Copy path replacements.

        self.__best_model_path = None      # Best network model save path.
        self.__best_state_path = None      # Best execution save path.
        self.__last_model_path = None      # Last network model save path.
        self.__last_state_path = None      # Last execution save path.
        self.__progress_table_path = None  # Statistics table path.
        self.__progress_plot_path = None   # Progress graph plot path.
        self.__param_save_path = None      # Used parameter file save path.
        self.__overrides_save_path = None  # Parameter override save path.
        self.__data_save_path = None       # Data configuration file save path.
        self.__repo_save_path = None       # Repository information save path.
        self.__log_file_path = None        # Log file path.
        self.__archive_path = None         # Archive save path.

        # Set the parameters that are overwritten with the override valued instead of merging.
        #
        self.__dict_overwrites = {'patch shapes', 'label map', 'label ratios', 'categories'}

        # Initialize logging.
        #
        self.__configureexperimentid(experiment_name=experiment_name)
        self.__initlogging(progress_dir_path=progress_dir_path, file_log_level=file_log_level, continue_experiment=continue_experiment)

        # Process the configured parameters.
        #
        self.__configuresettings(data_file_path=data_file_path,
                                 data_overrides=data_overrides,
                                 data_copy_config=data_copy_config,
                                 param_file_path=param_file_path,
                                 param_overrides=param_overrides,
                                 create_stats=create_stats,
                                 continue_experiment=continue_experiment,
                                 random_seed=random_seed,
                                 cpu_count_enforce=cpu_count_enforce)

        self.__configurepaths(progress_dir_path=progress_dir_path, archive_dir_path=archive_dir_path, work_dir_path=work_dir_path, caller_dir_paths=caller_dir_paths)

    def __configureexperimentid(self, experiment_name):
        """
        Configure the experiment identifier.

        Args:
            experiment_name (str): Experiment name.
        """

        # Construct experiment identifier.
        #
        self.__experiment_name = experiment_name
        self.__experiment_id = experiment_name if experiment_name else datetime.datetime.now().strftime('experiment_%Y-%m-%d_%H-%M')

    def __configuresettings(self, data_file_path, data_overrides, data_copy_config, param_file_path, param_overrides, create_stats, continue_experiment, random_seed, cpu_count_enforce):
        """
        Save settings and overrides.

        Args:
            data_file_path (str): Path of the data source file to load.
            data_overrides (dict, None): Path overrides in data source.
            data_copy_config (dict, None): Data copy configuration: replacement for copy source.
            param_file_path (str): Path of the parameters file to load.
            param_overrides (dict, None): Value overrides in parameter configuration.
            create_stats (bool): Allow missing stat files, and create them if necessary.
            continue_experiment (bool): Whether to continue with a previously trained network.
            random_seed (int, None): Random seed for reproducible experiments.
            cpu_count_enforce (int, None): Number of CPUs to assume to be available for the experiment.
        """

        # Save the input configuration file paths and override dictionaries.
        #
        self.__data_file_path = data_file_path
        self.__data_overrides = data_overrides
        self.__data_copy_config = data_copy_config
        self.__param_file_path = param_file_path
        self.__param_overrides = param_overrides
        self.__create_stats = create_stats
        self.__continue_experiment = continue_experiment
        self.__random_seed = random_seed
        self.__cpu_count_enforce = cpu_count_enforce

        # Log all arguments.
        #
        self.__logger.debug('Arguments/Experiment name: {name}'.format(name=self.__experiment_name))
        self.__logger.debug('Arguments/Progress dir path: {path}'.format(path=self.__progress_dir_path))
        self.__logger.debug('Arguments/Work dir path: {path}'.format(path=self.__work_dir_path))
        self.__logger.debug('Arguments/Archive dir path: {path}'.format(path=self.__archive_dir_path))
        self.__logger.debug('Arguments/Data file path: {path}'.format(path=self.__data_file_path))
        self.__logger.debug('Arguments/Data overrides: {map}'.format(map=self.__data_overrides))
        self.__logger.debug('Arguments/Data copy: {map}'.format(map=self.__data_copy_config))
        self.__logger.debug('Arguments/Param file path: {path}'.format(path=self.__param_file_path))
        self.__logger.debug('Arguments/Param overrides: {map}'.format(map=self.__param_overrides))
        self.__logger.debug('Arguments/Create stats: {flag}'.format(flag=self.__create_stats))
        self.__logger.debug('Arguments/Continue experiment: {flag}'.format(flag=self.__continue_experiment))
        self.__logger.debug('Arguments/Random seed: {seed}'.format(seed=self.__random_seed))
        self.__logger.debug('Arguments/CPU count enforced: {count}'.format(count=self.__cpu_count_enforce))
        self.__logger.debug('Arguments/Caller dir paths: {paths}'.format(paths=self.__caller_dir_paths))

    def __configurepaths(self, progress_dir_path, archive_dir_path, work_dir_path, caller_dir_paths):
        """
        Configure the various output paths based on the target directories and the experiment identifier.

        Args:
            progress_dir_path (str): Progress data dump directory.
            archive_dir_path (str, None): Zip all outputs here after the training is done.
            work_dir_path (str, None): Work directory path.
            caller_dir_paths (list): List of source directory paths of the caller library for repository information.
        """

        # Save the parameters.
        #
        self.__progress_dir_path = progress_dir_path
        self.__archive_dir_path = archive_dir_path
        self.__work_dir_path = work_dir_path
        self.__caller_dir_paths = caller_dir_paths

        # Construct paths.
        #
        self.__best_model_path = os.path.join(progress_dir_path, '{id}_best_model.net'.format(id=self.__experiment_id))
        self.__best_state_path = os.path.join(progress_dir_path, '{id}_best_state.dat'.format(id=self.__experiment_id))
        self.__last_model_path = os.path.join(progress_dir_path, '{id}_last_model.net'.format(id=self.__experiment_id))
        self.__last_state_path = os.path.join(progress_dir_path, '{id}_last_state.dat'.format(id=self.__experiment_id))
        self.__progress_table_path = os.path.join(progress_dir_path, '{id}_progress.csv'.format(id=self.__experiment_id))
        self.__progress_plot_path = os.path.join(progress_dir_path, '{id}_progress.png'.format(id=self.__experiment_id))
        self.__param_save_path = os.path.join(progress_dir_path, '{id}_parameters{ext}'.format(id=self.__experiment_id, ext=os.path.splitext(self.__param_file_path)[1]))
        self.__overrides_save_path = os.path.join(progress_dir_path, '{id}_overrides.yaml'.format(id=self.__experiment_id))
        self.__data_save_path = os.path.join(progress_dir_path, '{id}_data{ext}'.format(id=self.__experiment_id, ext=os.path.splitext(self.__data_file_path)[1]))
        self.__repo_save_path = os.path.join(progress_dir_path, '{id}_repo.yaml'.format(id=self.__experiment_id))
        self.__archive_path = os.path.join(archive_dir_path, '{base}.zip'.format(base=self.__experiment_id)) if archive_dir_path else None

        # Log paths.
        #
        self.__logger.debug('Best network model path: {path}'.format(path=self.__best_model_path))
        self.__logger.debug('Best state configuration path: {path}'.format(path=self.__best_state_path))
        self.__logger.debug('Last network model path: {path}'.format(path=self.__last_model_path))
        self.__logger.debug('Last state configuration path: {path}'.format(path=self.__last_state_path))
        self.__logger.debug('Progress table path: {path}'.format(path=self.__progress_table_path))
        self.__logger.debug('Progress plot path: {path}'.format(path=self.__progress_plot_path))
        self.__logger.debug('Parameters save path: {path}'.format(path=self.__param_save_path))
        self.__logger.debug('Overrides save path: {path}'.format(path=self.__overrides_save_path))
        self.__logger.debug('Data save path: {path}'.format(path=self.__data_save_path))
        self.__logger.debug('Repository info path: {path}'.format(path=self.__repo_save_path))
        self.__logger.debug('Log file path: {path}'.format(path=self.__log_file_path))
        self.__logger.debug('Archive path: {path}'.format(path=self.__archive_path))

    def __initlogging(self, progress_dir_path, file_log_level, continue_experiment):
        """
        Initialize logging.

        Args:
            progress_dir_path (str): Progress data dump directory.
            file_log_level (str):
            continue_experiment (bool): Whether to continue with a previously trained network.
        """

        # Calculate the log file path.
        #
        self.__log_file_path = os.path.join(progress_dir_path, '{id}_log.txt'.format(id=self.__experiment_id))

        # Inject a file logger to the root.
        #
        root_logger = logging.getLogger(name=None)

        # Add file log handler if not present already.
        #
        if not any(isinstance(log_handler, logging.FileHandler) and log_handler.baseFilename == self.__log_file_path for log_handler in root_logger.handlers):
            file_log_level_code = logging.getLevelName(level=file_log_level.upper())
            file_log_entry_format = '%(asctime)s %(levelname)s %(module)s: %(message)s'
            file_log_mode = 'a' if continue_experiment else 'w'
            file_log_formatter = logging.Formatter(fmt=file_log_entry_format)
            file_log_handler = logging.FileHandler(filename=self.__log_file_path, mode=file_log_mode)
            file_log_handler.setFormatter(fmt=file_log_formatter)
            file_log_handler.setLevel(level=file_log_level_code)

            root_logger.addHandler(hdlr=file_log_handler)

        # Configure logging. This class relies on configured logging somewhere down on the hierarchy.
        #
        qualified_class_name = '{module_name}.{class_name}'.format(module_name=self.__class__.__module__, class_name=self.__class__.__name__)
        self.__logger = logging.getLogger(name=qualified_class_name)

        # The first entry is the experiment identifier.
        #
        self.__logger.info('Experiment: {id}'.format(id=self.__experiment_id))

    def __flushlogging(self):
        """Flush all log handlers."""

        for log_handler in self.__logger.handlers:
            log_handler.flush()

    def __saveconfigfiles(self):
        """Save parameter and data configuration paths to the given location."""

        shutil.copyfile(src=self.__param_file_path, dst=self.__param_save_path)
        shutil.copyfile(src=self.__data_file_path, dst=self.__data_save_path)

    def __saveoverrides(self):
        """Save the data path and parameter value overrides to a single file."""

        # Combine the two overrides into a single dictionary.
        #
        overrides_map = {'data path': self.__data_overrides if self.__data_overrides is not None else {}, 'parameter value': self.__param_overrides if self.__param_overrides is not None else {}}

        with open(file=self.__overrides_save_path, mode='w') as overrides_file:
            yaml.dump(data=overrides_map, stream=overrides_file, indent=4, default_flow_style=False)

    def __repositoryversions(self):
        """
        Collect the repository versions of the used repositories: DigitalPathology and from the ones in the list.

        Returns:
            dict: Basic repository information structure.
        """

        # Get the DigitalPathology repository information through the BatchGenerator. The origin is 3 steps down from the root of the repository.
        #
        dpt_repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(dptbatchgenerator.__file__)))

        # Create repository information list and add the DigitalPathology repository information.
        #
        repo_dir_paths = [dpt_repo_dir] + (self.__caller_dir_paths if self.__caller_dir_paths is not None else [])

        repo_info_map = {}
        for repo_index in range(len(repo_dir_paths)):
            current_repo_path = repo_dir_paths[repo_index]
            current_repo_info = dptgitrepo.git_info(repo_dir_path=current_repo_path)
            if current_repo_info:
                repo_info_map[current_repo_info.get('name', repo_index)] = {info_key: info_value for info_key, info_value in current_repo_info.items() if info_key != 'name'}
                repo_info_map[current_repo_info.get('name', repo_index)]['path'] = current_repo_path

        # Return  the collected information.
        #
        return repo_info_map

    def __saverepositoryinfo(self):
        """Save the repository information list to a YAML file."""

        # Collect repository information.
        #
        repository_info = self.__repositoryversions()

        # Log experiment name, repository information and file paths.
        #
        for repository_name, repository_details in repository_info.items():
            self.__logger.info('Repository: {name} ({branch}) {revision}'.format(name=repository_name, branch=repository_details.get('branch'), revision=repository_details.get('revision')))
            self.__logger.debug('Repository: {name} from {url} at {path}'.format(name=repository_name, url=repository_details.get('url'), path=repository_details.get('path')))

        with open(file=self.__repo_save_path, mode='w') as repo_file:
            yaml.dump(data=repository_info, stream=repo_file, indent=4, default_flow_style=False)

    def __mergeparameters(self, parameters, overrides):
        """
        Merge parameter overrides with the parameter dictionaries.

        Args:
            parameters (dict): Parameter values from the configuration file.
            overrides (dict): Parameter value overrides.

        Returns:
            dict: Merged parameter values.
        """

        # Copy the dictionary to be extended to the result.
        #
        updated_parameters = parameters.copy()

        # Merge the extender dictionary with the original.
        #
        for key in updated_parameters:
            if key in overrides:
                if type(updated_parameters[key]) == dict and key not in self.__dict_overwrites:
                    updated_parameters[key] = self.__mergeparameters(updated_parameters[key], overrides[key])
                else:
                    updated_parameters[key] = overrides[key]

        # Return the result dictionary.
        #
        return updated_parameters

    def __seedrandomizers(self):
        """Seed the random number generators."""

        # Pick a random seed: use the configured value or generate one.
        #
        random_seed = self.__random_seed if self.__random_seed is not None else int(time.time() * 10000000)

        # Log the random seed for reproducibility.
        #
        self.__logger.info('System random seed: {seed}'.format(seed=random_seed))

        # Initialize the random number generators. If the seed is None, the system time is used as seed.
        #
        random.seed(a=random_seed, version=2)
        np.random.seed(seed=random.randint(a=0, b=np.iinfo(np.uint32).max))

    def __loadconfiguration(self):
        """
        Load the configuration file.

        Returns:
            dict, dict, dict, dict, dict: Model, system, training, data and augmentation parameters.
        """

        # Load parameters.
        #
        self.__logger.info('Loading configuration: {path}'.format(path=self.__param_file_path))

        with open(file=self.__param_file_path, mode='r') as param_file:
            parameters = yaml.load(stream=param_file, Loader=yaml.SafeLoader)

        # Apply parameter value overrides.
        #
        if self.__param_overrides:
            parameters = self.__mergeparameters(parameters=parameters, overrides=self.__param_overrides)

        # Disassemble the configuration structure into model, data, system and training parts.
        #
        return parameters['model'], parameters['system'], parameters['training'], parameters['data'], parameters['augmentation']

    def __loaddatasource(self, data_config):
        """
        Load data sources to a data source object.

        Args:
            data_config (dict): Dictionary of data configuration parameters.

        Returns:
            dptbatchsource.BatchSource: Batch source.

        Raises:
            ValueError: The purpose ratios in the parameter configuration file and data source file do not match.

            DigitalPathologyDataError: Data errors.
            DigitalPathologyConfigError: Configuration errors.
        """

        # Log parameters.
        #
        self.__logger.info('Loading data source: {path}'.format(path=self.__data_file_path))

        self.__logger.debug('Data/Purposes: {distribution}'.format(distribution=data_config['purposes']))
        self.__logger.debug('Data/Categories: {distribution}'.format(distribution=data_config['categories']))

        # Create the batch source object first to make it parse the configuration YAML file.
        #
        batch_source = dptbatchsource.BatchSource(source_items=None)
        batch_source.load(file_path=self.__data_file_path)
        if self.__data_overrides:
            batch_source.update(path_replacements=self.__data_overrides)

        # Validate the actual purpose distribution from the data file against the distribution from the configuration file.
        #
        if batch_source.purposes():
            # Validate the two purpose distributions.
            #
            if not batch_source.validate(purpose_distribution=data_config['purposes']):
                self.__logger.error('Configuration file - data file purpose distribution mismatch: {config} - {data}'.format(config=data_config['purposes'], data=batch_source.distribution()))
                raise ValueError('Configuration file - data file purpose distribution mismatch: {config} - {data}'.format(config=data_config['purposes'], data=batch_source.distribution()))
        else:
            # The data file is not distributed yet.
            #
            batch_source.distribute(purpose_distribution=data_config['purposes'])

        # Log parameters.
        #
        self.__logger.info('Source items: {count}'.format(count=batch_source.count(purpose_id=['training', 'validation'])))

        self.__logger.debug('Source/Source items: {count}'.format(count=batch_source.count(purpose_id=None, category_id=None)))
        self.__logger.debug('Source/Purposes: {purposes}'.format(purposes=batch_source.purposes()))
        self.__logger.debug('Source/Categories: {categories}'.format(categories=batch_source.categories()))

        # Return the configured batch source.
        #
        return batch_source

    def __constructnetworkmodel(self, network_model, model_config, data_config):
        """
        Construct network model object.

        Args:
            network_model (ModelBase, None): Optional configured network to use for training.
            model_config (dict): Dictionary of network model parameters.
            data_config (dict): Dictionary of data configuration parameters.

        Returns:
            diagmodels.models.modelbase.ModelBase: Constructed model.

        Raises:
            ValueError: Unknown backend type.
            ValueError: Unknown model type.
            ValueError: The model is not implemented on the selected backend.
            NotImplementedError: Missing model implementation on the selected backend.

            DigitalPathologyModelError: Model errors.
        """

        model = None

        # Check if a new network shall be constructed or an existing one should be used.
        #
        if network_model:
            # Log network name for identification.
            #
            self.__logger.info('Using pre-configured network model: {name}'.format(name=network_model.name))

            model = network_model
        else:
            # Determine class count.
            #
            class_count = len(set(data_config['labels']['label map'].values()))

            # Log parameters.
            #
            self.__logger.info('Configuring network model: {model_type}'.format(model_type=model_config['type']))

            self.__logger.debug('Model/Type: {model_type}'.format(model_type=model_config['type']))
            self.__logger.debug('Model/Name: {model_name}'.format(model_name=model_config['name']))
            self.__logger.debug('Model/Description: {model_description}'.format(model_description=model_config['description']))
            self.__logger.debug('Model/Batch norm: {batch_norm}'.format(batch_norm=model_config['batch norm']))
            self.__logger.debug('Model/Dropout layers: {count}'.format(count=model_config['dropout layers']))
            self.__logger.debug('Model/Dropout probability: {prob}'.format(prob=model_config['dropout probability']))
            self.__logger.debug('Model/Branching factor: {factor}'.format(factor=model_config['branching factor']))
            self.__logger.debug('Model/L2 lambda: {l2_lambda}'.format(l2_lambda=model_config['L2 lambda']))
            self.__logger.debug('Model/Channels first: {channels_first}'.format(channels_first=model_config['channels first']))

            self.__logger.debug('Data/Images/Patch shapes: {shape}'.format(shape=data_config['images']['patch shapes']))
            self.__logger.debug('Data/Labels/Label count: {count}'.format(count=class_count))

            # Select the patch shape.
            #
            spacing = list(data_config['images']['patch shapes'].keys())[0]

            if model_config['backend'] == 'lasagne':
                patch_shape = (len(data_config['images']['channels']),) + tuple(data_config['images']['patch shapes'][spacing])
            elif model_config['backend'] == 'keras':
                patch_shape = tuple(data_config['images']['patch shapes'][spacing]) + (len(data_config['images']['channels']),)
            elif model_config['backend'] == 'pytorch':
                patch_shape = (len(data_config['images']['channels']),) + tuple(data_config['images']['patch shapes'][spacing])
            else:
                # The given backend type is unknown.
                #
                self.__logger.error('Unknown backend type: \'{backend_type}\''.format(backend_type=model_config['backend']))
                raise ValueError('Unknown backend type: \'{backend_type}\''.format(backend_type=model_config['backend']))

            # Check model type and construct a network accordingly.
            #
            if model_config['type'] == 'naturenet':
                # Local imports.
                #
                if model_config['backend'] == 'lasagne':
                    import diagmodels.models.lasagne.naturenet as dmsnaturenetmodel
                elif model_config['backend'] == 'keras':
                    import diagmodels.models.keras.naturenet as dmsnaturenetmodel

                else:
                    # The given backend type is unknown.
                    #
                    self.__logger.error('Unknown backend type: \'{backend_type}\''.format(backend_type=model_config['backend']))
                    raise ValueError('Unknown backend type: \'{backend_type}\''.format(backend_type=model_config['backend']))

                # Construct network.
                #
                model = dmsnaturenetmodel.NatureNet(name=model_config['name'], description=model_config['description'])

                # Configure parameters.
                #
                model.configure(input_shape=patch_shape,
                                classes=class_count,
                                branching_factor=model_config['branching factor'],
                                batch_norm=model_config['batch norm'],
                                dropout_count=model_config['dropout layers'],
                                dropout_prob=model_config['dropout probability'],
                                l2_lambda=model_config['L2 lambda'],
                                channels_first=model_config['channels first'])

            elif model_config['type'] == 'naturenet256v1':
                # Local imports.
                #
                if model_config['backend'] == 'lasagne':
                    import diagmodels.models.lasagne.naturenet256v1 as dmsnaturenet256v1model
                elif model_config['backend'] == 'keras':
                    raise NotImplementedError('The naturenet256v1 model not implemented on Keras.')
                else:
                    # The given backend type is unknown.
                    #
                    self.__logger.error('Unknown backend type: \'{backend_type}\''.format(backend_type=model_config['backend']))
                    raise ValueError('Unknown backend type: \'{backend_type}\''.format(backend_type=model_config['backend']))

                # Construct network.
                #
                model = dmsnaturenet256v1model.NatureNet256v1(name=model_config['name'], description=model_config['description'])

                # Configure parameters.
                #
                model.configure(input_shape=patch_shape,
                                classes=class_count,
                                branching_factor=model_config['branching factor'],
                                batch_norm=model_config['batch norm'],
                                dropout_count=model_config['dropout layers'],
                                dropout_prob=model_config['dropout probability'],
                                l2_lambda=model_config['L2 lambda'],
                                channels_first=model_config['channels first'])

            elif model_config['type'] == 'unet':
                # Log the model specific parameters.
                #
                self.__logger.debug('Model/Depth: {depth}'.format(depth=model_config['depth']))
                self.__logger.debug('Model/Padding: {padding}'.format(padding=model_config['padding']))

                if model_config['backend'] == 'lasagne':
                    # Local imports.
                    #
                    import diagmodels.models.lasagne.unet as dmsunetmodel

                    # Construct network.
                    #
                    model = dmsunetmodel.UNet(name=model_config['name'], description=model_config['description'])

                    # Configure parameters.
                    #
                    model.configure(input_shape=patch_shape,
                                    depth=model_config['depth'],
                                    classes=class_count,
                                    branching_factor=model_config['branching factor'],
                                    batch_norm=model_config['batch norm'],
                                    dropout_count=model_config['dropout layers'],
                                    dropout_prob=model_config['dropout probability'],
                                    l2_lambda=model_config['L2 lambda'],
                                    padding=model_config['padding'],
                                    channels_first=model_config['channels first'])

                elif model_config['backend'] == 'keras':
                    # Log backend specific parameters.
                    #
                    self.__logger.debug('Model/Upsampling: {upsampling}'.format(upsampling=model_config['upsampling']))
                    self.__logger.debug('Model/Downsampling: {downsampling}'.format(downsampling=model_config['downsampling']))

                    # Local imports.
                    #
                    import diagmodels.models.keras.unet as dmsunetmodel

                    # Construct network.
                    #
                    model = dmsunetmodel.UNet(name=model_config['name'], description=model_config['description'])

                    # Configure parameters.
                    #
                    model.configure(input_shape=patch_shape,
                                    depth=model_config['depth'],
                                    classes=class_count,
                                    branching_factor=model_config['branching factor'],
                                    batch_norm=model_config['batch norm'],
                                    dropout_count=model_config['dropout layers'],
                                    dropout_prob=model_config['dropout probability'],
                                    l2_lambda=model_config['L2 lambda'],
                                    padding=model_config['padding'],
                                    residual=model_config['residual'],
                                    downsampling=model_config['downsampling'],
                                    upsampling=model_config['upsampling'],
                                    channels_first=model_config['channels first'],
                                    loss=model_config['loss'])

            elif model_config['type'] == 'deeplab':
                if model_config['backend'] == 'pytorch':
                    # Local imports.
                    #
                    import diagmodels.models.pytorch.deeplabnet as dmsdeeplabmodel

                    # Construct network.
                    #
                    model = dmsdeeplabmodel.DeepLabNet(name=model_config['name'], description=model_config['description'])

                    # Configure parameters.
                    #
                    model.configure(input_shape=patch_shape,
                                    classes=class_count,
                                    backbone=model_config['backbone'])

                else:
                    # The given backend type is unknown.
                    #
                    self.__logger.error('Unknown backend type: \'{backend_type}\''.format(backend_type=model_config['backend']))
                    raise ValueError('Unknown backend type: \'{backend_type}\''.format(backend_type=model_config['backend']))

            elif model_config['type'] == 'inception':
                # Incpetion is only implemented in Keras.
                #
                if model_config['backend'] == 'keras':
                    # Local imports.
                    #
                    import diagmodels.models.keras.inception_v3 as dmsinceptionmodel

                    # Construct network.
                    #
                    model = dmsinceptionmodel.InceptionV3(name=model_config['name'], description=model_config['description'])

                    # Configure parameters.
                    #
                    model.configure(input_shape=patch_shape,
                                    classes=class_count,
                                    include_top=model_config['include top'],
                                    preload_weights=model_config['preload weights'],
                                    use_aux_classifier=model_config['use aux classifier'],
                                    depth_multiplier=model_config['depth multiplier'],
                                    min_depth=model_config['minimum depth'],
                                    channels_first=model_config['channels first'])
                else:
                    # The given backend type is unknown.
                    #
                    self.__logger.error('Invalid backend type: \'{backend_type}\' for the model:  \'{model_type}\''.format(backend_type=model_config['backend'], model_type=model_config['type']))
                    raise ValueError('Invalid backend type: \'{backend_type}\' for the model:  \'{model_type}\''.format(backend_type=model_config['backend'], model_type=model_config['type']))

            elif model_config['type'] == 'inception_v4':
                # Incpetion is only implemented in Keras.
                #
                if model_config['backend'] == 'keras':
                    # Local imports.
                    #
                    import diagmodels.models.keras.inception_v4 as dmsinceptionmodel

                    # Construct network.
                    #
                    model = dmsinceptionmodel.Inceptionv4(name=model_config['name'], description=model_config['description'])

                    # Configure parameters.
                    #
                    model.configure(input_shape=patch_shape,
                                    classes=class_count,
                                    load_weights=model_config['preload weights'],
                                    dropout_prob=model_config['dropout probability'],
                                    l2_lambda=model_config['L2 lambda'],
                                    channels_first=model_config['channels first'])
                else:
                    # The given backend type is unknown.
                    #
                    self.__logger.error('Invalid backend type: \'{backend_type}\' for the model:  \'{model_type}\''.format(backend_type=model_config['backend'], model_type=model_config['type']))
                    raise ValueError('Invalid backend type: \'{backend_type}\' for the model:  \'{model_type}\''.format(backend_type=model_config['backend'], model_type=model_config['type']))

            elif model_config['type'] == 'fcnn':
                # Incpetion is only implemented in Keras.
                #
                if model_config['backend'] == 'keras':
                    # Log the model specific parameters.
                    #
                    self.__logger.debug('Model/Depth: {depth}'.format(depth=model_config['depth']))

                    # Local imports.
                    #
                    import diagmodels.models.keras.general_cnn as dmsfcnn

                    # Construct network.
                    #
                    model = dmsfcnn.general_FCNN(name=model_config['name'], description=model_config['description'])

                    # Configure parameters.
                    #
                    model.configure(input_shape=patch_shape,
                                    depth=model_config['depth'],
                                    classes=class_count,
                                    branching_factor=model_config['branching factor'],
                                    batch_norm=model_config['batch norm'],
                                    dropout_count=model_config['dropout layers'],
                                    dropout_prob=model_config['dropout probability'],
                                    l2_lambda=model_config['L2 lambda'],
                                    channels_first=model_config['channels first'])
                else:
                    # The given backend type is unknown.
                    #
                    self.__logger.error('Invalid backend type: \'{backend_type}\' for the model:  \'{model_type}\''.format(backend_type=model_config['backend'], model_type=model_config['type']))
                    raise ValueError('Invalid backend type: \'{backend_type}\' for the model:  \'{model_type}\''.format(backend_type=model_config['backend'], model_type=model_config['type']))

            else:
                # The given model type is unknown.
                #
                self.__logger.error('Unknown model type: \'{model_type}\''.format(model_type=model_config['type']))
                raise ValueError('Unknown model type: \'{model_type}\''.format(model_type=model_config['type']))

        # Log the input dimension order for the model.
        #
        self.__logger.debug('Model dimension order: \'{dimension_order}\''.format(dimension_order=model.dimensionorder.upper()))

        # Return the configured network. It is still not built, just configured.
        #
        return model

    def __constructaugmenters(self, augmentation_config):
        """
        Construct patch augmenters.

        Args:
            augmentation_config (dict): Dictionary of augmentation configuration parameters.

        Returns:
            dptaugmenterpool.AugmenterPool, dptaugmenterpool.AugmenterPool: Training and validation patch augmenters.

        Raises:
            ValueError: Unknown augmentation purpose.
            ValueError: Unknown augmentation type.

            AugmentationGroupAlreadyExistsError: Augmentation group already exists.
            UnknownAugmentationGroupError: Unknown augmentation group.
            InvalidAugmentationRatioError: The ratio of the augmentation selection in its group is invalid.

            InvalidContrastSigmaRangeError: The contrast adjustment range is not valid.
            InvalidHaematoxylinSigmaRangeError: The sigma range for Haematoxylin channel adjustment is not valid.
            InvalidHaematoxylinBiasRangeError: The bias range for Haematoxylin channel adjustment is not valid.
            InvalidEosinSigmaRangeError: The sigma range for Eosin channel adjustment is not valid.
            InvalidEosinBiasRangeError: The bias range for Eosin channel adjustment is not valid.
            InvalidDabSigmaRangeError: The sigma range for DAB channel adjustment is not valid.
            InvalidDabBiasRangeError: The bias range for DAB channel adjustment is not valid.
            InvalidCutoffRangeError: The cutoff range is not valid.
            InvalidHueSigmaRangeError: The sigma range for Hue channel adjustment is not valid.
            InvalidSaturationSigmaRangeError: The sigma range for Saturation channel adjustment is not valid.
            InvalidBrightnessSigmaRangeError: The sigma range for Brightness channel adjustment is not valid.
            InvalidAdditiveGaussianNoiseSigmaRangeError: The sigma range for additive Gaussian noise is not valid.
            InvalidBlurSigmaRangeError: The sigma range for Gaussian blur is not valid.
            InvalidElasticSigmaIntervalError: The interval of sigma for elastic deformation is invalid.
            InvalidElasticAlphaIntervalError: The interval of alpha for elastic deformation is invalid.
            InvalidElasticMapCountError: The number of elastic deformation maps to precalculate is invalid.
            InvalidElasticInterpolationOrderError: The interpolation order for elastic transformation is not valid.
            InvalidFlipListError: The flip list is invalid.
            InvalidRotationRepetitionListError: The list for 90 degree rotation repetition is invalid.
            InvalidScalingRangeError: The sigma range for scaling is not valid.
            InvalidScalingInterpolationOrderError: The interpolation order for scaling is not valid.
        """

        self.__logger.info('Configuring augmenters')

        # Prepare the return values.
        #
        training_augmenter = None
        validation_augmenter = None

        # Go through the purposes: training or validation.
        #
        for purpose_id, purpose_config in augmentation_config.items():
            # Log augmentation summary.
            #
            augmentation_summary = []
            for group_item in purpose_config:
                if group_item['random']:
                    items = {augmenter_item['type']: augmenter_item['ratio'] for augmenter_item in group_item['items']}
                else:
                    items = [augmenter_item['type'] for augmenter_item in group_item['items']]

                augmentation_summary.append({'group': group_item['group'], 'items': items})

            config_str = '; '.join('\'{group}\': {items}'.format(group=summary_item['group'], items=summary_item['items']) for summary_item in augmentation_summary)

            self.__logger.info('Augmentation for {purpose}: {config}'.format(purpose=purpose_id, config=config_str))

            # Initialize augmenter pool.
            #
            patch_augmenter = dptaugmenterpool.AugmenterPool()

            # Instantiate and add each augmenter from the configuration to the pool.
            #
            for group_item in purpose_config:
                # Add augmentation group.
                #
                self.__logger.debug('Augmentation; group: \'{group}\'; random: {mode}'.format(group=group_item['group'], mode=group_item['random']))

                patch_augmenter.appendgroup(group=group_item['group'], randomized=group_item['random'])

                # Add all augmenter items in the group.
                #
                for augmenter_item in group_item['items']:
                    # Get the ratio if the group is randomized.
                    #
                    current_ratio = augmenter_item['ratio'] if group_item['random'] else 0.0
                    ratio_log = 'ratio: {ratio}'.format(ratio=augmenter_item['ratio']) if group_item['random'] else 'sequential'

                    # Create augmenter object.
                    #
                    if augmenter_item['type'] == 'contrast':
                        # Contrast enhancement patch augmentation.
                        #
                        current_augmenter = dptcontrastaugmenter.ContrastAugmenter(sigma_range=augmenter_item['sigma'])

                        self.__logger.debug('Augmentation; group: \'{group}\'; type: Contrast adjustment; {ratio}; id: \'{id}\'; sigma range: {sigma}'.format(group=group_item['group'],
                                                                                                                                                              ratio=ratio_log,
                                                                                                                                                              id=augmenter_item['type'],
                                                                                                                                                              sigma=augmenter_item['sigma']))
                    elif augmenter_item['type'] == 'hed_color':
                        # Saturation enhancement patch augmentation.
                        #
                        current_augmenter = dpthedcoloraugmenter.HedColorAugmenter(haematoxylin_sigma_range=augmenter_item['haematoxylin']['sigma'],
                                                                                   haematoxylin_bias_range=augmenter_item['haematoxylin']['bias'],
                                                                                   eosin_sigma_range=augmenter_item['eosin']['sigma'],
                                                                                   eosin_bias_range=augmenter_item['eosin']['bias'],
                                                                                   dab_sigma_range=augmenter_item['dab']['sigma'],
                                                                                   dab_bias_range=augmenter_item['dab']['bias'],
                                                                                   cutoff_range=augmenter_item['cutoff'])

                        log_message_sigmas = '{haematoxylin}, {eosin}, {dab}'.format(haematoxylin=augmenter_item['haematoxylin']['sigma'],
                                                                                     eosin=augmenter_item['eosin']['sigma'],
                                                                                     dab=augmenter_item['dab']['sigma'])

                        log_message_biases = '{haematoxylin}, {eosin}, {dab}'.format(haematoxylin=augmenter_item['haematoxylin']['bias'],
                                                                                     eosin=augmenter_item['eosin']['bias'],
                                                                                     dab=augmenter_item['dab']['bias'])

                        log_message = 'Augmentation; group: \'{group}\'; type: HED adjustment; {ratio}; id: \'{id}\'; HED sigma ranges: {sigmas}; HED bias ranges: {biases}; Cut-off range: {cutoff}'
                        self.__logger.debug(log_message.format(group=group_item['group'],
                                                               ratio=ratio_log,
                                                               id=augmenter_item['type'],
                                                               sigmas=log_message_sigmas,
                                                               biases=log_message_biases,
                                                               cutoff=augmenter_item['cutoff']))
                    elif augmenter_item['type'] == 'hsb_color':
                        # Saturation enhancement patch augmentation.
                        #
                        current_augmenter = dpthsbcoloraugmenter.HsbColorAugmenter(hue_sigma_range=augmenter_item['hue'],
                                                                                   saturation_sigma_range=augmenter_item['saturation'],
                                                                                   brightness_sigma_range=augmenter_item['brightness'])

                        log_message = 'Augmentation; group: \'{group}\'; type: HSB adjustment; {ratio}; id: \'{id}\'; HSV ranges: {hue}, {saturation}, {brightness}'
                        self.__logger.debug(log_message.format(group=group_item['group'],
                                                               ratio=ratio_log,
                                                               id=augmenter_item['type'],
                                                               hue=augmenter_item['hue'],
                                                               saturation=augmenter_item['saturation'],
                                                               brightness=augmenter_item['brightness']))
                    elif augmenter_item['type'] == 'additive':
                        # Additive Gaussian noise patch augmentation.
                        #
                        current_augmenter = dtpadditiveguassiannoiseaugmenter.AdditiveGaussianNoiseAugmenter(sigma_range=augmenter_item['sigma'])

                        self.__logger.debug('Augmentation; group: \'{group}\'; type: Additive Gaussian noise; {ratio}; id: \'{id}\'; sigma range: {sigma}'.format(group=group_item['group'],
                                                                                                                                                                  ratio=ratio_log,
                                                                                                                                                                  id=augmenter_item['type'],
                                                                                                                                                                  sigma=augmenter_item['sigma']))
                    elif augmenter_item['type'] == 'blur':
                        # Gaussian blur patch augmentation.
                        #
                        current_augmenter = dptgaussianbluraugmenter.GaussianBlurAugmenter(sigma_range=augmenter_item['sigma'])

                        self.__logger.debug('Augmentation; group: \'{group}\'; type: Gaussian blur; {ratio}; id: \'{id}\'; sigma range: {sigma}'.format(group=group_item['group'],
                                                                                                                                                        ratio=ratio_log,
                                                                                                                                                        id=augmenter_item['type'],
                                                                                                                                                        sigma=augmenter_item['sigma']))
                    elif augmenter_item['type'] == 'elastic':
                        # Elastic deformation patch augmentation.
                        #
                        current_augmenter = dptelasticagumenter.ElasticAugmenter(sigma_interval=augmenter_item['sigma'],
                                                                                 alpha_interval=augmenter_item['alpha'],
                                                                                 map_count=augmenter_item['maps'],
                                                                                 interpolation_order=augmenter_item['order'])

                        log_message = 'Augmentation; group: \'{group}\'; type: Elastic deformation; {ratio}; id: \'{id}\'; sigma range: {sigma}; alpha range: {alpha}; maps: {maps}; order: {order}'
                        self.__logger.debug(log_message.format(group=group_item['group'],
                                                               ratio=ratio_log,
                                                               id=augmenter_item['type'],
                                                               sigma=augmenter_item['sigma'],
                                                               alpha=augmenter_item['sigma'],
                                                               maps=augmenter_item['maps'],
                                                               order=augmenter_item['order']))
                    elif augmenter_item['type'] == 'flip':
                        # Flipping patch augmentation.
                        #
                        current_augmenter = dptflipaugmenter.FlipAugmenter(flip_list=augmenter_item['flips'])

                        self.__logger.debug('Augmentation; group: \'{group}\'; type: Flipping; {ratio}; id: \'{id}\'; flips: {flips}'.format(group=group_item['group'],
                                                                                                                                             ratio=ratio_log,
                                                                                                                                             id=augmenter_item['type'],
                                                                                                                                             flips=augmenter_item['flips']))
                    elif augmenter_item['type'] == 'rotate_90':
                        # Rotation by multiples of 90 degrees patch augmentation.
                        #
                        current_augmenter = dptrotate90augmenter.Rotate90Augmenter(k_list=augmenter_item['rotations'])

                        self.__logger.debug('Augmentation; group: \'{group}\'; type: Rotation by 90s; {ratio}; id: \'{id}\'; multiples: {multiples}'.format(group=group_item['group'],
                                                                                                                                                            ratio=ratio_log,
                                                                                                                                                            id=augmenter_item['type'],
                                                                                                                                                            multiples=augmenter_item['rotations']))
                    elif augmenter_item['type'] == 'scale':
                        # Scaling patch augmentation.
                        #
                        current_augmenter = dptscalingaugmenter.ScalingAugmenter(scaling_range=augmenter_item['scaling'], interpolation_order=augmenter_item['order'])

                        self.__logger.debug('Augmentation; group: \'{group}\'; type: Scaling; {ratio}; id: \'{id}\'; scaling: {scaling}; order: {order}'.format(group=group_item['group'],
                                                                                                                                                                ratio=ratio_log,
                                                                                                                                                                id=augmenter_item['type'],
                                                                                                                                                                scaling=augmenter_item['scaling'],
                                                                                                                                                                order=augmenter_item['order']))
                    elif augmenter_item['type'] == 'pass-through':
                        # Pass through patch augmenter.
                        #
                        current_augmenter = dptpassthroughaugmenter.PassThroughAugmenter()

                        self.__logger.debug('Augmentation; group: \'{group}\'; type: Pass through; {ratio}; id: \'{id}\''.format(group=group_item['group'],
                                                                                                                                 ratio=ratio_log,
                                                                                                                                 id=augmenter_item['type']))
                    else:
                        # The given augmentation type is unknown.
                        #
                        self.__logger.error('Unknown augmentation type: \'{augmentation_type}\''.format(augmentation_type=augmenter_item['type']))
                        raise ValueError('Unknown augmentation type: \'{augmentation_type}\''.format(augmentation_type=augmenter_item['type']))

                    # Append the augmenter object to the pool.
                    #
                    patch_augmenter.appendaugmenter(augmenter=current_augmenter, group=group_item['group'], ratio=current_ratio)

            # Store the augmenter pool for the right purpose.
            #
            if purpose_id == 'training':
                training_augmenter = patch_augmenter
            elif purpose_id == 'validation':
                validation_augmenter = patch_augmenter
            else:
                # The given augmentation purpose is unknown.
                #
                self.__logger.error('Unknown augmentation purpose: \'{augmentation_purpose}\''.format(augmentation_purpose=purpose_id))
                raise ValueError('Unknown augmentation purpose: \'{augmentation_purpose}\''.format(augmentation_purpose=purpose_id))

        # Return the configured patch augmenter pools.
        #
        return training_augmenter, validation_augmenter

    def __constructnormalizers(self, data_config):
        """
        Construct batch normalizers.
        Args:
            data_config (dict): Dictionary of data configuration parameters.

        Returns:
            dptnormalizerbase.NormalizerBase, dptnormalizerbase.NormalizerBase: Training and validation batch normalizers.

        Raises:
            ValueError: Unknown normalization type.

            InvalidNormalizationRangeError: Source or target normalization range is invalid.
        """

        self.__logger.info('Configuring normalizers')

        # Prepare the return values.
        #
        training_normalizer = None
        validation_normalizer = None

        # Log normalization summary.
        #
        self.__logger.info('Normalization: {config}'.format(config=data_config['normalization']))

        # Check if batch normalization enabled.
        #
        if data_config['normalization']['enabled']:
            # Configure the normalizer objects.
            #
            if data_config['normalization']['type'] == 'general':
                training_normalizer = dtpgeneralnormalizer.GeneralNormalizer(target_range=data_config['normalization']['target range'], source_range=data_config['normalization']['source range'])
                validation_normalizer = dtpgeneralnormalizer.GeneralNormalizer(target_range=data_config['normalization']['target range'], source_range=data_config['normalization']['source range'])

                self.__logger.debug('Normalization; type: {normalization_type}; source range: {src}; target range: {dst}'.format(normalization_type=data_config['normalization']['type'],
                                                                                                                                 src=data_config['normalization']['source range'],
                                                                                                                                 dst=data_config['normalization']['target range']))
            elif data_config['normalization']['type'] == 'rgb':
                training_normalizer = dtprgbnormalizer.RgbNormalizer(target_range=data_config['normalization']['target range'])
                validation_normalizer = dtprgbnormalizer.RgbNormalizer(target_range=data_config['normalization']['target range'])

                self.__logger.debug('Normalization; type: {normalization_type}; target range: {dst}'.format(normalization_type=data_config['normalization']['type'],
                                                                                                            dst=data_config['normalization']['target range']))
            elif data_config['normalization']['type'] == 'rgb_to_0-1':
                training_normalizer = dtprgbtozeroonenormalizer.RgbToZeroOneNormalizer()
                validation_normalizer = dtprgbtozeroonenormalizer.RgbToZeroOneNormalizer()

                self.__logger.debug('Normalization; type: {normalization_type}'.format(normalization_type=data_config['normalization']['type']))
            elif data_config['normalization']['type'] == 'pass-through':
                training_normalizer = dtppassthroughnormalizer.PassThroughNormalizer()
                validation_normalizer = dtppassthroughnormalizer.PassThroughNormalizer()

                self.__logger.debug('Normalization; type: {normalization_type}'.format(normalization_type=data_config['normalization']['type']))
            else:
                # The given normalization type is unknown.
                #
                self.__logger.error('Unknown normalization type: \'{normalization_type}\''.format(normalization_type=data_config['normalization']['type']))
                raise ValueError('Unknown normalization type: \'{normalization_type}\''.format(normalization_type=data_config['normalization']['type']))

        # Return the configured batch normalizer objects.
        #
        return training_normalizer, validation_normalizer

    def __constructlabelmappers(self, data_config):
        """
        Construct the label value to label index mappers.

        Args:
            data_config (dict): Dictionary of data configuration parameters.

        Returns:
            (dptlabelmapper.LabelMapper, dptlabelmapper.LabelMapper): Training and validation label mappers.

        Raises:
            DigitalPathologyLabelError: Label errors.
        """

        self.__logger.info('Configuring label mappers')

        # Log parameters.
        #
        self.__logger.debug('Data/Labels/Label map: \'{map}\''.format(map=data_config['labels']['label map']))

        # Return training and validation label mappers.
        #
        return dptlabelmapper.LabelMapper(label_map=data_config['labels']['label map']), dptlabelmapper.LabelMapper(label_map=data_config['labels']['label map'])

    def __constructweightmappers(self, data_config):
        """
        Construct the weight mapper objects.

        Args:
            data_config (dict): Dictionary of data configuration parameters.

        Returns:
            (dptweightmapperbase.WeightMapperBase, dptweightmapperbase.WeightMapperBase,
             dptweightmapperbase.WeightMapperBase,dptweightmapperbase.WeightMapperBase): Training and validation patch weight mappers and training and validation batch weight mappers.

        Raises:
            ValueError: Unknown weight mapping type.

            DigitalPathologyWeightMappingError: Weight mapping errors.
        """

        self.__logger.info('Configuring weight mappers')

        # Prepare the return values.
        #
        training_patch_weight_mapper = None
        training_batch_weight_mapper = None

        validation_patch_weight_mapper = None
        validation_batch_weight_mapper = None

        # Log label weight mapping.
        #
        self.__logger.info('Weight mapping enabled: {flag}'.format(flag=data_config['weight mapping']['enabled']))

        # Check if label weight mapping enabled.
        #
        if data_config['weight mapping']['enabled']:
            # Get the number of available classes.
            #
            class_count = len(set(data_config['labels']['label map'].values()))

            # Log label weight mapping configuration.
            #
            self.__logger.info('Weight mapping type: {config}; classes: {count}'.format(config=data_config['weight mapping']['type'], count=class_count))

            # Configure the weight mapper objects.
            #
            if data_config['weight mapping']['type'] == 'clean':
                training_patch_weight_mapper = dptcleanweightmapper.CleanWeightMapper(classes=class_count)
                validation_patch_weight_mapper = dptcleanweightmapper.CleanWeightMapper(classes=class_count)

            elif data_config['weight mapping']['type'] == 'balancing':
                training_patch_weight_mapper = dptpatchweightmapper.PatchWeightMapper(classes=class_count,
                                                                                      normalize=data_config['weight mapping']['normalize'],
                                                                                      clip_min=data_config['weight mapping']['clipping']['min'],
                                                                                      clip_max=data_config['weight mapping']['clipping']['max'])

                validation_patch_weight_mapper = dptpatchweightmapper.PatchWeightMapper(classes=class_count,
                                                                                        normalize=data_config['weight mapping']['normalize'],
                                                                                        clip_min=data_config['weight mapping']['clipping']['min'],
                                                                                        clip_max=data_config['weight mapping']['clipping']['max'])

                self.__logger.debug('Weight mapping; normalize: {flag}; clipping: [{clip_min}, {clip_max}]'.format(flag=data_config['weight mapping']['normalize'],
                                                                                                                   clip_min=data_config['weight mapping']['clipping']['min'],
                                                                                                                   clip_max=data_config['weight mapping']['clipping']['max']))
            elif data_config['weight mapping']['type'] == 'batch balancing':
                training_patch_weight_mapper = dptcleanweightmapper.CleanWeightMapper(classes=class_count)
                training_batch_weight_mapper = dptbatchweightmapper.BatchWeightMapper(classes=class_count,
                                                                                      normalize=data_config['weight mapping']['normalize'],
                                                                                      clip_min=data_config['weight mapping']['clipping']['min'],
                                                                                      clip_max=data_config['weight mapping']['clipping']['max'])

                validation_patch_weight_mapper = dptcleanweightmapper.CleanWeightMapper(classes=class_count)
                validation_batch_weight_mapper = dptbatchweightmapper.BatchWeightMapper(classes=class_count,
                                                                                        normalize=data_config['weight mapping']['normalize'],
                                                                                        clip_min=data_config['weight mapping']['clipping']['min'],
                                                                                        clip_max=data_config['weight mapping']['clipping']['max'])

            else:
                # The given label weight mapping type is unknown.
                #
                self.__logger.error('Unknown weight mapping type: \'{mapping_type}\''.format(mapping_type=data_config['weight mapping']['type']))
                raise ValueError('Unknown weight mapping type: \'{mapping_type}\''.format(mapping_type=data_config['weight mapping']['type']))

        # Return the configured weight mapper objects.
        #
        return training_patch_weight_mapper, validation_patch_weight_mapper, training_batch_weight_mapper, validation_batch_weight_mapper

    def __initializedatasource(self, data_source):
        """
        Load data sources to a data source object, copy the files or check their existence.

        Args:
            data_source (dptbatchsource.BatchSource): Data source.

        Raises:
            ValueError: Non existing files in the configuration.
        """

        self.__logger.info('Initializing data source')

        # Copy the data to the target location if configured or check if there is no copy directive.
        #
        purposes = ['training', 'validation']

        if self.__data_copy_config is not None:
            copied_items, skipped_items = dptimagefile.copy_batch_source(batch_source=data_source,
                                                                         source_replacements=self.__data_copy_config,
                                                                         target_replacements=None,
                                                                         purposes=purposes,
                                                                         categories=None,
                                                                         allow_missing_stat=self.__create_stats,
                                                                         overwrite=False)

            # Report item counts.
            #
            self.__logger.info('Copied items: {count}'.format(count=copied_items))
            self.__logger.info('Skipped items: {count}'.format(count=skipped_items))

        else:
            okay_items, missing_items = dptimagefile.check_batch_source(batch_source=data_source,
                                                                        purposes=purposes,
                                                                        categories=None,
                                                                        allow_missing_stat=self.__create_stats)

            # Report item counts.
            #
            self.__logger.info('Checked items: {count}'.format(count=okay_items))
            self.__logger.info('Missing items: {count}'.format(count=missing_items))

            if 0 < missing_items:
                self.__logger.error('Missing items from data source: {count}'.format(count=missing_items))
                raise ValueError('Missing items from data source: {count}'.format(count=missing_items))

            # Collect the directories to create.
            #
            if self.__create_stats:
                directories_to_create = set(os.path.dirname(source_item.stat) for source_item in data_source.items(purpose_id=purposes, category_id=None, replace=True) if source_item.stat)
                for directory_path in directories_to_create:
                    if os.path.isdir(directory_path):
                        self.__logger.debug('Directory exists: \'{path}\''.format(path=directory_path))
                    else:
                        self.__logger.debug('Create directory: \'{path}\''.format(path=directory_path))
                        os.makedirs(directory_path, exist_ok=True)

    def __constructgenerators(self,
                              data_source,
                              training_normalizer,
                              validation_normalizer,
                              training_augmenter,
                              validation_augmenter,
                              training_label_mapper,
                              validation_label_mapper,
                              training_patch_weight_mapper,
                              validation_patch_weight_mapper,
                              training_batch_weight_mapper,
                              validation_batch_weight_mapper,
                              data_config,
                              system_config,
                              training_config,
                              dimension_order):
        """
        Configure batch generators for training and validation.

        Args:
            data_source (dptbatchsource.BatchSource): Batch source.
            training_normalizer (dptnormalizerbase.NormalizerBase): Batch normalizer for training.
            validation_normalizer (dptnormalizerbase.NormalizerBase): Batch normalizer for validation.
            training_augmenter (dptaugmenterpool.AugmenterPool, None): Patch augmenter for training.
            validation_augmenter (dptaugmenterpool.AugmenterPool, None): Patch augmenter for validation.
            training_label_mapper (dptlabelmapper.LabelMapper): Label mapper for training.
            validation_label_mapper (dptlabelmapper.LabelMapper): Label mapper for validation.
            training_patch_weight_mapper (dptweightmapperbase.WeightMapperBase): Weight map calculator for training.
            validation_patch_weight_mapper (dptweightmapperbase.WeightMapperBase): Weight map calculator for validation.
            training_batch_weight_mapper: (dptweightmapperbase.WeightMapperBase): Batch based weight map calculator for training.
            validation_batch_weight_mapper: (dptweightmapperbase.WeightMapperBase): Batch based weight map calculator for validation.
            data_config (dict): Dictionary of data configuration parameters.
            system_config (dict): Dictionary of system configuration parameters.
            training_config (dict): Dictionary of training parameters.
            dimension_order (str): The dimension order that will be returned by the batch-gen.

        Returns:
            (dptbatchgenerator.BatchGenerator, dptbatchgenerator.BatchGenerator): Training and validation batch generators.

        Raises:
            ValueError: Invalid patch shape configuration.
            ValueError: The CPU count setting is invalid.

            DigitalPathologyBufferError: Buffer errors.
            DigitalPathologyConfigError: Configuration errors.
            DigitalPathologyDataError: Data errors.
            DigitalPathologyImageError: Image errors.
            DigitalPathologyLabelError: Label errors.
            DigitalPathologyProcessError: Process errors.
            DigitalPathologyWeightMappingError: Weight mapping errors.
        """

        # Log parameters.
        #
        self.__logger.info('Configuring batch generators')

        self.__logger.debug('Data/Images/Patch shapes: {shapes}'.format(shapes=data_config['images']['patch shapes']))
        self.__logger.debug('Data/Spacing tolerance: {tolerance}'.format(tolerance=data_config['spacing tolerance']))
        self.__logger.debug('Data/Images/Channels: {channels}'.format(channels=data_config['images']['channels']))
        self.__logger.debug('Data/Labels/Mask pixel spacing: {spacing}'.format(spacing=data_config['labels']['mask pixel spacing']))
        self.__logger.debug('Data/Labels/Label mode: \'{mode}\''.format(mode=data_config['labels']['label mode']))
        self.__logger.debug('Data/Labels/Label ratios: {distribution}'.format(distribution=data_config['labels']['label ratios']))
        self.__logger.debug('Data/Labels/Strict selection: {flag}'.format(flag=data_config['labels']['strict selection']))
        self.__logger.debug('Data/Categories: {distribution}'.format(distribution=data_config['categories']))
        self.__logger.debug('Data/Resources/Workers: {distribution}'.format(distribution=data_config['resources']['workers']))
        self.__logger.debug('Data/Resources/Samplers: {distribution}'.format(distribution=data_config['resources']['samplers']))
        self.__logger.debug('System/Process count: {count}'.format(count=system_config['process count']))
        self.__logger.debug('System/Pool size: {size}'.format(size=system_config['pool size']))
        self.__logger.debug('System/Multi-threaded: {multi}'.format(multi=system_config['multi-threaded']))
        self.__logger.debug('System/IPC chunk size: {size}'.format(size=system_config['ipc chunk size']))
        self.__logger.debug('System/Join timeout: {secs} sec'.format(secs=system_config['join timeout secs']))
        self.__logger.debug('System/Response timeout: {secs} sec'.format(secs=system_config['response timeout secs']))
        self.__logger.debug('System/Poll timeout: {secs} sec'.format(secs=system_config['poll timeout secs']))

        self.__logger.debug('Generator dimension order: \'{dimension_order}\''.format(dimension_order=dimension_order.upper()))

        # Check if the patch shapes are configured correctly with pixel spacing and not level index.
        #
        if any(type(shape) is not float for shape in data_config['images']['patch shapes']):
            raise ValueError('Invalid patch shape configuration, level index present instead of pixel spacing: {source}'.format(source=data_config['images']['patch shapes']))

        # Check CPU count setting.
        #
        if self.__cpu_count_enforce is not None and self.__cpu_count_enforce < 1:
            raise ValueError('Invalid CPU count setting: {count}'.format(count=self.__cpu_count_enforce))

        # Calculate process and pool size distributions.
        #
        sum_resource_dist = sum(data_config['resources']['samplers'].values())
        norm_resource_dist = {purpose_id: data_config['resources']['samplers'][purpose_id] / sum_resource_dist for purpose_id in data_config['resources']['samplers']}

        process_count_config = system_config['process count']
        cpu_count = self.__cpu_count_enforce if self.__cpu_count_enforce is not None else os.cpu_count()
        process_count = round(cpu_count * process_count_config) if type(process_count_config) == float else process_count_config

        if 0 < process_count:
            process_dist_config = {purpose_id: (norm_resource_dist[purpose_id], 1, process_count) for purpose_id in norm_resource_dist}
            process_count_dist = dptpopulation.distribute_population(population=process_count, ratios=process_dist_config)
        else:
            process_count_dist = {purpose_id: 0 for purpose_id in data_config['resources']['samplers']}

        sum_purpose_dist = sum(data_config['resources']['workers'].values())
        norm_purpose_dist = {purpose_id: data_config['resources']['workers'][purpose_id] / sum_purpose_dist for purpose_id in data_config['resources']['workers']}

        total_source_count = data_source.count(purpose_id=list(data_config['purposes'].keys()), category_id=list(data_config['categories'].keys()))
        if total_source_count < system_config['pool size']:
            self.__logger.warning('Pool size: {pool} is larger than the total number of available images: {count}'.format(pool=system_config['pool size'], count=total_source_count))

        pool_dist_config = {purpose_id: (norm_purpose_dist[purpose_id], 1, data_source.count(purpose_id=purpose_id, category_id=None)) for purpose_id in norm_purpose_dist}
        pool_size_count_maxed = min(total_source_count, system_config['pool size'])
        pool_size_dist = dptpopulation.distribute_population(population=pool_size_count_maxed, ratios=pool_dist_config)

        # Calculate the buffer size for training and validation.
        #
        training_main_buffer_size = training_config['iterations']['training']['iteration count'] * training_config['iterations']['training']['batch size']
        validation_buffer_size = training_config['iterations']['validation']['iteration count'] * training_config['iterations']['validation']['batch size']

        # Log CPU usage.
        #
        process_count_config_str = '/'.join([str(process_count_item) for process_count_item in process_count_dist.values()])
        self.__logger.info('Available CPUs: {count}'.format(count=cpu_count))
        self.__logger.info('Process count: {processes}'.format(processes=process_count_config_str))
        self.__logger.debug('Process distribution: {distribution}'.format(distribution=process_count_dist))
        self.__logger.debug('Pool distribution: {distribution}'.format(distribution=pool_size_dist))

        # Check process count against sampler count.
        #
        if pool_size_dist['training'] < process_count_dist['training']:
            self.__logger.warning('CPU count for training: {cpu} is larger than the patch sampler pool size: {sampler}'.format(cpu=process_count_dist['training'],
                                                                                                                               sampler=pool_size_dist['training']))

        if pool_size_dist['validation'] < process_count_dist['validation']:
            self.__logger.warning('CPU count for validation: {cpu} is larger than the patch sampler pool size: {sampler}'.format(cpu=process_count_dist['validation'],
                                                                                                                                 sampler=pool_size_dist['validation']))

        # Log derived values.
        #
        self.__logger.debug('Training main buffer size: {size}'.format(size=training_main_buffer_size))
        self.__logger.debug('Training read buffer size: {size}'.format(size=training_main_buffer_size))
        self.__logger.debug('Validation main buffer size: {size}'.format(size=validation_buffer_size))
        self.__logger.debug('Validation read buffer size: {size}'.format(size=validation_buffer_size))

        # Check if all the categories in the batch source are configured for use.
        #
        if set(data_config['categories'].keys()) != set(data_source.categories(purpose_id='training')):
            training_categories = sorted(data_source.categories(purpose_id='training'))
            config_categories = sorted(data_config['categories'].keys())
            self.__logger.warning('Not all categories in the training data source: {data} are configured for sampling in the data configuration: {config}'.format(data=training_categories,
                                                                                                                                                                  config=config_categories))

        if set(data_config['categories'].keys()) != set(data_source.categories(purpose_id='validation')):
            validation_categories = sorted(data_source.categories(purpose_id='validation'))
            config_categories = sorted(data_config['categories'].keys())
            self.__logger.warning('Not all categories in the validation data source: {data} are configured for sampling in the data configuration: {config}'.format(data=validation_categories,
                                                                                                                                                                    config=config_categories))

        # Create batch generator for training.
        #
        training_batch_generator = dptbatchgenerator.BatchGenerator(label_dist=data_config['labels']['label ratios'],
                                                                    patch_shapes=data_config['images']['patch shapes'],
                                                                    mask_spacing=data_config['labels']['mask pixel spacing'],
                                                                    spacing_tolerance=data_config['spacing tolerance'],
                                                                    input_channels=data_config['images']['channels'],
                                                                    dimension_order=dimension_order,
                                                                    label_mode=data_config['labels']['label mode'],
                                                                    patch_sources=data_source.collection(purpose_id='training', category_id=None, replace=True),
                                                                    category_dist=data_config['categories'],
                                                                    strict_selection=data_config['labels']['strict selection'],
                                                                    create_stats=self.__create_stats,
                                                                    main_buffer_size=training_main_buffer_size,
                                                                    buffer_chunk_size=training_config['buffer chunk size'],
                                                                    read_buffer_size=training_main_buffer_size,
                                                                    labels_one_hot=data_config['labels']['one hot'],
                                                                    batch_normalizer=training_normalizer,
                                                                    patch_augmenter=training_augmenter,
                                                                    label_mapper=training_label_mapper,
                                                                    free_label_range=False,
                                                                    patch_weight_mapper=training_patch_weight_mapper,
                                                                    batch_weight_mapper=training_batch_weight_mapper,
                                                                    multi_threaded=system_config['multi-threaded'],
                                                                    sampler_process_count=process_count_dist['training'],
                                                                    sampler_pool_size=pool_size_dist['training'],
                                                                    sampler_chunk_size=system_config['ipc chunk size'],
                                                                    join_timeout=system_config['join timeout secs'],
                                                                    response_timeout=system_config['response timeout secs'],
                                                                    poll_timeout=system_config['poll timeout secs'],
                                                                    name_tag='training')

        # Create batch generator for validation.
        #
        validation_batch_generator = dptbatchgenerator.BatchGenerator(label_dist=data_config['labels']['label ratios'],
                                                                      patch_shapes=data_config['images']['patch shapes'],
                                                                      mask_spacing=data_config['labels']['mask pixel spacing'],
                                                                      spacing_tolerance=data_config['spacing tolerance'],
                                                                      input_channels=data_config['images']['channels'],
                                                                      dimension_order=dimension_order,
                                                                      label_mode=data_config['labels']['label mode'],
                                                                      patch_sources=data_source.collection(purpose_id='validation', category_id=None, replace=True),
                                                                      category_dist=data_config['categories'],
                                                                      strict_selection=data_config['labels']['strict selection'],
                                                                      create_stats=self.__create_stats,
                                                                      main_buffer_size=validation_buffer_size,
                                                                      buffer_chunk_size=training_config['buffer chunk size'],
                                                                      read_buffer_size=validation_buffer_size,
                                                                      labels_one_hot=data_config['labels']['one hot'],
                                                                      batch_normalizer=validation_normalizer,
                                                                      patch_augmenter=validation_augmenter,
                                                                      label_mapper=validation_label_mapper,
                                                                      free_label_range=False,
                                                                      patch_weight_mapper=validation_patch_weight_mapper,
                                                                      batch_weight_mapper=validation_batch_weight_mapper,
                                                                      multi_threaded=system_config['multi-threaded'],
                                                                      sampler_process_count=process_count_dist['validation'],
                                                                      sampler_pool_size=pool_size_dist['validation'],
                                                                      sampler_chunk_size=system_config['ipc chunk size'],
                                                                      join_timeout=system_config['join timeout secs'],
                                                                      response_timeout=system_config['response timeout secs'],
                                                                      poll_timeout=system_config['poll timeout secs'],
                                                                      name_tag='validation')

        # Return the created generators. They are still in an invalid state.
        #
        return training_batch_generator, validation_batch_generator

    def __constructfilesynchronizer(self):
        """
        Configure file synchronizer object.

        Returns:
            dptfilesynchronizer.FileSynchronizer: File synchronizer object.

        Raises:
            DigitalPathologyTrainingError: Training errors.
        """

        self.__logger.info('Configuring file synchronizer')

        # Create the work directory if necessary.
        #
        if self.__work_dir_path:
            os.makedirs(self.__work_dir_path, exist_ok=True)

        # Construct synchronizer object. If the work directory is not configured the synchronization will be disabled and output will be written to directly to the target paths.
        #
        file_synchronizer = dptfilesynhcronizer.FileSynchronizer(work_directory=self.__work_dir_path)

        # Add all file paths.
        #
        file_synchronizer.add(target_path=self.__progress_table_path)
        file_synchronizer.add(target_path=self.__progress_plot_path)
        file_synchronizer.add(target_path=self.__best_model_path)
        file_synchronizer.add(target_path=self.__best_state_path)
        file_synchronizer.add(target_path=self.__last_model_path)
        file_synchronizer.add(target_path=self.__last_state_path)

        # Return the configured synchronizer object.
        #
        return file_synchronizer

    def __constructstataggregator(self, file_synchronizer):
        """
        Configure statistics aggregator.

        Args:
            file_synchronizer (dptfilesynchronizer.FileSynchronizer): File synchronizer object.

        Returns:
            dptstats.StatAggregator: Configured statistics aggregator.
        """

        self.__logger.info('Configuring statistics aggregator')

        # Calculate output paths.
        #
        progress_table_path = file_synchronizer.work(target_path=self.__progress_table_path)
        progress_plot_path = file_synchronizer.work(target_path=self.__progress_plot_path)

        self.__logger.debug('Progress table work path: {path}'.format(path=progress_table_path))
        self.__logger.debug('Progress plot work path: {path}'.format(path=progress_plot_path))

        # Instantiate stat aggregator.
        #
        stat_aggregator = dptstats.StatAggregator(epoch_save_path=progress_table_path,
                                                  epoch_plot_path=progress_plot_path,
                                                  epoch_stats_to_plot=[],
                                                  experiment_name=self.__experiment_id,
                                                  append=self.__continue_experiment)

        # Return configured aggregator.
        #
        return stat_aggregator

    def __executetraining(self, network_model, training_generator, validation_generator, stat_aggregator, file_synchronizer, training_config):
        """
        Execute the training process with the configured parameters.

        Args:
            network_model (dptnaturenetmodel.NatureNetModel): Network model.
            training_generator (dptbatchgenerator.BatchGenerator): Training batch generator.
            validation_generator (dptbatchgenerator.BatchGenerator): Validation batch generator.
            stat_aggregator (dptstats.StatAggregator): Configured statistics aggregator.
            file_synchronizer (dptfilesynchronizer.FileSynchronizer): File synchronizer object.
            training_config (dict): Dictionary of training parameters.

        Raises:
            InvalidIterationLogPercentError: The iteration log percent is out of (0.0, 1.0] bounds.
            InvalidNetworkObjectError: The network object is invalid.
            InvalidBatchGeneratorError: The batch generator object is invalid.
            DimensionOrderMismatchError: Model - generator dimension order mismatch.
            InvalidStatsHandlerError: The statistics handler is invalid.
            InvalidEpochCountError: The epoch count in less than 1.
            InvalidRepetitionCountError: The repetition count is less than 1.
            InvalidIterationCountError: The training or validation iteration count is less than 1.
            InvalidBufferConfigurationError: Boosting is enabled without double buffering.
            InvalidDifficultThreshold: The difficult example threshold is out of the [0.0, 1.0] interval.
            InvalidDifficultUpdateRatio: The difficult example update ratio is out of the [0.0, 1.0] interval.
            InvalidLearningRateError: Non positive learning rate.
            InvalidModelSavePathError: Invalid network model dump path.
            InvalidStateSavePathError: Invalid training state dump path.

            DigitalPathologyBufferError: Buffer errors.
            DigitalPathologyConfigError: Configuration errors.
            DigitalPathologyDataError: Data errors.
            DigitalPathologyImageError: Image errors.
            DigitalPathologyLabelError: Label errors.
            DigitalPathologyModelError: Model errors.
            DigitalPathologyProcessError: Process errors.
            DigitalPathologyStatError: Stats errors.
        """

        # Log parameters.
        #
        self.__logger.info('Starting...')

        self.__logger.debug('Training/Epoch count: {count}'.format(count=training_config['epoch count']))
        self.__logger.debug('Training/Metric name: {name}'.format(name=training_config['metric name']))
        self.__logger.debug('Training/Higher is better: {flag}'.format(flag=training_config['higher is better']))
        self.__logger.debug('Training/Averaging length: {length}'.format(length=training_config['averaging length']))
        self.__logger.debug('Training/Source step length: {length}'.format(length=training_config['source step length']))
        self.__logger.debug('Training/Iteration log percent: {percent}'.format(percent=training_config['iteration log percent']))
        self.__logger.debug('Training/Buffer chunk size: {epoch}'.format(epoch=training_config['buffer chunk size']))
        self.__logger.debug('Training/Learning/Learning rate: {rate}'.format(rate=training_config['learning']['learning rate']))
        self.__logger.debug('Training/Learning/Learning rate decay/Enabled: {flag}'.format(flag=training_config['learning']['learning rate decay']['enabled']))
        self.__logger.debug('Training/Learning/Learning rate decay/Update factor: {factor}'.format(factor=training_config['learning']['learning rate decay']['update factor']))
        self.__logger.debug('Training/Learning/Learning rate decay/Plateau length: {plateau}'.format(plateau=training_config['learning']['learning rate decay']['plateau length']))
        self.__logger.debug('Training/Learning/Stop plateau/Enabled: {flag}'.format(flag=training_config['learning']['stop plateau']['enabled']))
        self.__logger.debug('Training/Learning/Stop plateau/Plateau length: {plateau}'.format(plateau=training_config['learning']['stop plateau']['plateau length']))
        self.__logger.debug('Training/Boosting/Enabled: {flag}'.format(flag=training_config['boosting']['enabled']))
        self.__logger.debug('Training/Boosting/Buffer mode switch: {epoch}'.format(epoch=training_config['boosting']['buffer mode switch']))
        self.__logger.debug('Training/Boosting/Difficult threshold: {threshold}'.format(threshold=training_config['boosting']['difficult threshold']))
        self.__logger.debug('Training/Boosting/Difficult update ratio: {ratio}'.format(ratio=training_config['boosting']['difficult update ratio']))
        self.__logger.debug('Training/Iterations/Training/Repetition count: {repetitions}'.format(repetitions=training_config['iterations']['training']['repetition count']))
        self.__logger.debug('Training/Iterations/Training/Iteration count: {iters}'.format(iters=training_config['iterations']['training']['iteration count']))
        self.__logger.debug('Training/Iterations/Training/Batch size: {size}'.format(size=training_config['iterations']['training']['batch size']))
        self.__logger.debug('Training/Iterations/Validation/Repetition count: {repetitions}'.format(repetitions=training_config['iterations']['validation']['repetition count']))
        self.__logger.debug('Training/Iterations/Validation/Iteration count: {iters}'.format(iters=training_config['iterations']['validation']['iteration count']))
        self.__logger.debug('Training/Iterations/Validation/Batch size: {size}'.format(size=training_config['iterations']['validation']['batch size']))

        # Calculate output paths.
        #
        best_model_path = file_synchronizer.work(target_path=self.__best_model_path)
        best_state_path = file_synchronizer.work(target_path=self.__best_state_path)
        last_model_path = file_synchronizer.work(target_path=self.__last_model_path)
        last_state_path = file_synchronizer.work(target_path=self.__last_state_path)

        self.__logger.debug('Best network model work path: {path}'.format(path=best_model_path))
        self.__logger.debug('Best state configuration work path: {path}'.format(path=best_state_path))
        self.__logger.debug('Last network model work path: {path}'.format(path=last_model_path))
        self.__logger.debug('Last state configuration work path: {path}'.format(path=last_state_path))

        # Instantiate a network trainer.
        #
        trainer = dptnetworktrainer.NetworkTrainer(model=network_model,
                                                   training_batch_generator=training_generator,
                                                   validation_batch_generator=validation_generator,
                                                   stat_aggregator=stat_aggregator,
                                                   file_synchronizer=file_synchronizer,
                                                   metric_name=training_config['metric name'],
                                                   higher_is_better=training_config['higher is better'],
                                                   averaging_length=training_config['averaging length'],
                                                   iter_log_percent=training_config['iteration log percent'])

        # Execute training with the trainer.
        #
        trainer.execute(epoch_count=training_config['epoch count'],
                        source_step_length=training_config['source step length'],
                        training_repetition_count=training_config['iterations']['training']['repetition count'],
                        validation_repetition_count=training_config['iterations']['validation']['repetition count'],
                        training_iter_count=training_config['iterations']['training']['iteration count'],
                        validation_iter_count=training_config['iterations']['validation']['iteration count'],
                        training_batch_size=training_config['iterations']['training']['batch size'],
                        validation_batch_size=training_config['iterations']['validation']['batch size'],
                        boosting_enabled=training_config['boosting']['enabled'],
                        buffer_mode_switch=training_config['boosting']['buffer mode switch'],
                        difficult_threshold=training_config['boosting']['difficult threshold'],
                        difficult_update_ratio=training_config['boosting']['difficult update ratio'],
                        learning_rate=training_config['learning']['learning rate'],
                        learning_rate_decay_enabled=training_config['learning']['learning rate decay']['enabled'],
                        learning_rate_update_factor=training_config['learning']['learning rate decay']['update factor'],
                        learning_rate_update_plateau=training_config['learning']['learning rate decay']['plateau length'],
                        stop_plateau_enabled=training_config['learning']['stop plateau']['enabled'],
                        stop_plateau_length=training_config['learning']['stop plateau']['plateau length'],
                        best_model_path=best_model_path,
                        best_state_path=best_state_path,
                        last_model_path=last_model_path,
                        last_state_path=last_state_path,
                        continue_experiment=self.__continue_experiment)

    def __archivefiles(self, file_synchronizer):
        """
        Move all files in the list to the target archive file.

        Args:
            file_synchronizer (dptfilesynchronizer.FileSynchronizer, None): File synchronizer object.
        """

        # Archiving is not mandatory.
        #
        if self.__archive_path:
            # Initialize ZIP.
            #
            zip_file = zipfile.ZipFile(file=self.__archive_path, mode='w', compression=zipfile.ZIP_DEFLATED, allowZip64=True)

            # Construct list of all files.
            #
            file_path_list = [self.__best_model_path,
                              self.__best_state_path,
                              self.__last_model_path,
                              self.__last_state_path,
                              self.__progress_table_path,
                              self.__progress_plot_path,
                              self.__param_save_path,
                              self.__overrides_save_path,
                              self.__data_save_path,
                              self.__repo_save_path,
                              self.__log_file_path]

            # Add each file to the ZIP.
            #
            for file_path in file_path_list:
                actual_file_path = file_synchronizer.work(target_path=file_path) if file_synchronizer is not None else file_path
                if actual_file_path is not None and os.path.isfile(actual_file_path):
                    zip_file.write(filename=actual_file_path, arcname=os.path.basename(file_path))

            # Close the ZIP to release the source files.
            #
            zip_file.close()

    @property
    def experiment(self):
        """
        Get the identifier of the experiment used by this object.

        Returns:
            str: Experiment identifier.
        """

        return self.__experiment_id

    def execute(self, network=None):
        """
        Execute the configured experiment.

        Args:
            network (ModelBase, None): Optional configured network to use for training.

        Raises:
            DigitalPathologyError: Digital pathology error.
        """

        # Save the parameters and the data source, parameter and path overrides, and repository information.
        #
        self.__saveconfigfiles()
        self.__saveoverrides()
        self.__saverepositoryinfo()

        file_synchronizer = None

        try:
            # Seed the random number generators.
            #
            self.__seedrandomizers()

            # Load the configuration file.
            #
            model_config, system_config, training_config, data_config, augmentation_config = self.__loadconfiguration()

            # Load data source.
            #
            data_source = self.__loaddatasource(data_config=data_config)

            # Create one of the default networks if no configured network was passed to the commander.
            #
            network_model = self.__constructnetworkmodel(network_model=network, model_config=model_config, data_config=data_config)

            # Configure patch augmenters.
            #
            training_augmenter, validation_augmenter = self.__constructaugmenters(augmentation_config=augmentation_config)

            # Construct batch normalizers.
            #
            training_normalizer, validation_normalizer = self.__constructnormalizers(data_config=data_config)

            # Configure label mappers.
            #
            training_label_mapper, validation_label_mapper = self.__constructlabelmappers(data_config=data_config)

            # Configure weight map calculators.
            #
            training_weight_mapper, validation_weight_mapper, training_batch_weight_mapper, validation_batch_weight_mapper = self.__constructweightmappers(data_config=data_config)

            # Configure batch generators.
            #
            training_generator, validation_generator = self.__constructgenerators(data_source=data_source,
                                                                                  training_normalizer=training_normalizer,
                                                                                  validation_normalizer=validation_normalizer,
                                                                                  training_augmenter=training_augmenter,
                                                                                  validation_augmenter=validation_augmenter,
                                                                                  training_label_mapper=training_label_mapper,
                                                                                  validation_label_mapper=validation_label_mapper,
                                                                                  training_patch_weight_mapper=training_weight_mapper,
                                                                                  validation_patch_weight_mapper=validation_weight_mapper,
                                                                                  training_batch_weight_mapper=training_batch_weight_mapper,
                                                                                  validation_batch_weight_mapper=validation_batch_weight_mapper,
                                                                                  data_config=data_config,
                                                                                  system_config=system_config,
                                                                                  training_config=training_config,
                                                                                  dimension_order=network_model.dimensionorder)

            # Initialize file synchronizer.
            #
            file_synchronizer = self.__constructfilesynchronizer()

            # Initialize stats handler.
            #
            stat_aggregator = self.__constructstataggregator(file_synchronizer=file_synchronizer)

            # Initialize data: copy or check the source files.
            #
            self.__initializedatasource(data_source=data_source)

            # Execute training.
            #
            self.__executetraining(network_model=network_model,
                                   training_generator=training_generator,
                                   validation_generator=validation_generator,
                                   stat_aggregator=stat_aggregator,
                                   file_synchronizer=file_synchronizer,
                                   training_config=training_config)

        except Exception as exception:
            # Collect and summarize traceback information.
            #
            _, _, exception_traceback = sys.exc_info()
            trace_string = dpttrace.format_traceback(traceback_object=exception_traceback)

            # Log the exception.
            #
            self.__logger.info('Exception raised: "{ex}"'.format(ex=exception))
            self.__logger.error('Exception: "{ex}"; trace: "{trace}"'.format(ex=exception, trace=trace_string))

            # Re-raise the exception.
            #
            raise

        finally:
            # Flush all log content.
            #
            self.__logger.info('Archiving results...')
            self.__flushlogging()

            # Move result files to the archive path.
            #
            self.__archivefiles(file_synchronizer=file_synchronizer)
