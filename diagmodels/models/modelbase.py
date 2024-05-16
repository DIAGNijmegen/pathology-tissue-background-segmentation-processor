"""
Base class for network models.
"""

from ..errors import modelerrors as dptmodelerrors

from abc import ABCMeta, abstractmethod
import os
import importlib
import pickle
import zlib
#import git
import re

#----------------------------------------------------------------------------------------------------

class ModelBase(object, metaclass=ABCMeta):
    """This class is the base class for all network model classes."""

    def __init__(self, name=None, description=None):
        """
        Initialize model parameters.

        Args:
            name (str): Name of the model.
            description (str): Description of the model.

        Raises:
            InvalidImageLevel: The configured level is negative.
        """

        # Initialize base class.
        #
        super().__init__()

        # Initialize members.
        #
        self.__name = name                # Name of the network model.
        self.__description = description  # Description of the network model.
        self._model_instance = None       # Instance of model in backend
        self._compiled = False

    def __classinfo(self):
        """
        Get the name of this class

        Returns:
            str: Name of this class.
        """

        return '{module_name}.{class_name}'.format(module_name=self.__class__.__module__, class_name=self.__class__.__name__)

    def __repositoryinfo(self):
        """
        Collect the repository version of DiagModels.

        Returns:
            dict: Basic repository information structure.
        """

        # Build a repository object and obtain basic information: name, URL, revision, and branch.
        #
        try:
            repo_dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
            repo = git.Repo(repo_dir_path)
        except:
            return None
        else:
            repo_url = repo.remotes.origin.url
            repo_name_rev = repo.head.object.name_rev

        # Initialize info dictionary.
        #
        repository_info = {'url': repo_url}

        # Find the name of the repository in the URL.
        #
        repo_name_match = re.match(pattern='.*/(?P<name>.*)(\\.git)?$', string=repo_url)
        if repo_name_match:
            repository_info['name'] = repo_name_match.group('name')

            # Find the hed revision of the repository and the name of the active branch.
            #
            repo_revision_branch_match = re.match(pattern='^(?P<revision>[0-9a-f]*) (?P<branch>.*)$', string=repo_name_rev)
            if repo_revision_branch_match:
                repository_info['revision'] = repo_revision_branch_match.group('revision')
                repository_info['branch'] = repo_revision_branch_match.group('branch')

        return repository_info

    def _customdata(self):
        """
        Get a map of custom data structure to add to the dictionary of exported values. This function should be redefined in subclasses.

        Returns:
            dict: Dictionary of values to add to the exported data.
        """

        # By default it returns an empty map.
        #
        return {}

    def _setcustomdata(self, data_maps):
        """
        Set the class specific data. This function should be redefined in subclasses.

        Args:
            data_maps (dict): Custom data map.
        """

        pass

    @abstractmethod
    def _restoremodelparameters(self, parameters):
        """
        Restores the state of the model and the optimizer

        Args:
            parameters: Dictionary of parameters
        """

        pass

    @abstractmethod
    def _modelparameters(self):
        """
        Collect the parameters of the network to save.

        Returns:
            dict: Dictionary of parameters.
        """

        return {}

    @property
    def name(self):
        """
        Name of the model.

        Returns:
            str: Name of the model.
        """

        return self.__name

    @property
    def compiled(self):
        """
        Compilation state

        Returns:
            bool: Whether the model's current state has been compiled
        """
        return self._compiled

    @property
    def description(self):
        """
        Description of the model.

        Returns:
            str: Description of the model.
        """

        return self.__description

    @property
    def model(self):
        """
        Get the backend model instance.

        Returns:
            Model instance or None.
        """

        # Return the network instance.
        #
        return self._model_instance

    @abstractmethod
    def build(self):
        """Build the network instance with the pre-configured parameters."""

        pass

    @staticmethod
    def instantiate(file):
        """
        Load the network file, investigate the class of the stored object and instantiate the right network object that can load the stored data.

        Args:
            file (str, dict): Model file path or loaded data structure.

        Returns:
            ModelBase: Model object.

        Raises:
            InvalidModelDataFormatError: Invalid model data format.
        """

        # If the file is a string, it is used as a file path to load. Otherwise it is expected to contain the already loaded model data.
        #
        if type(file) is str:
            with open(file=file, mode='rb') as file:
                model_dump = pickle.loads(zlib.decompress(file.read()))
        elif type(file) is dict:
            model_dump = file
        else:
            # Unknown model data format.
            #
            raise dptmodelerrors.InvalidModelDataFormatError()

        # Determine the module and the class of the model.
        #
        model_class_elements = model_dump['class'].split('.')
        model_module_str = '.'.join(model_class_elements[:-1])
        model_class_str = model_class_elements[-1]

        model_module = importlib.import_module(model_module_str)
        model_class = getattr(model_module, model_class_str)

        # Instantiate the model and load the data.
        #
        model = model_class()
        model.load(file=model_dump)

        # Return the prepared model.
        #
        return model

    def load(self, file):
        """
        Load network model and parameters from file or already loaded data structure.

        Args:
            file (str, dict): Model file path or the loaded data dictionary.

        Raises:
            InvalidModelDataFormatError: Invalid model data format.
            ModelClassError: The loaded data does not match the class of the model.
        """

        # If the file is a string, it is used as a file path to load. Otherwise it is expected to contain the already loaded model data.
        #
        if type(file) is str:
            with open(file=file, mode='rb') as file:
                model_dump = pickle.loads(zlib.decompress(file.read()))
        elif type(file) is dict:
            model_dump = file
        else:
            # Unknown model data format.
            #
            raise dptmodelerrors.InvalidModelDataFormatError()

        if self.__classinfo() == model_dump['class']:
            self.__name = model_dump['name']
            self.__description = model_dump['description']

            self._setcustomdata(model_dump['data'])
            self._restoremodelparameters(model_dump['parameters'])
        else:
            # The model class in the loaded data does not match the class of this object.
            #
            raise dptmodelerrors.ModelClassError(self.__classinfo(), model_dump['class'])

    def save(self, file_path):
        """
        Save network model and parameters to file.

        Args:
            file_path (str): Data file path.
        """

        # Construct network dump.
        #
        model_dump = {'name': self.__name,
                      'description': self.__description,
                      'class': self.__classinfo(),
                      'revision': self.__repositoryinfo(),
                      'data': self._customdata(),
                      'parameters': self._modelparameters()}

        # Dump the data to file.
        #
        with open(file=file_path, mode='wb') as file:
            file.write(zlib.compress(pickle.dumps(model_dump), level=9))

    @abstractmethod
    def unfix(self):
        """Unfix input shape."""

        pass

    def clear(self):
        """Clear the network instance."""

        # Clear everything.
        #
        self._model_instance = None

    @abstractmethod
    def update(self, x, y, *args, **kwargs):
        """
        Update the network.

        Args:
            x (numpy.ndarray or list of numpy.ndarray): an array or list of arrays containing training data.
            y (numpy.ndarray or list of numpy.ndarray): an array or list of arrays containing label data.
            *args (list): Ordered arguments.
            **kwargs (dict): Keyword arguments.

        Returns:
            dict: Output of the update function.
        """

        return {}

    @abstractmethod
    def validate(self, x, y, *args, **kwargs):
        """
        Validate the network.

        Args:
            x (numpy.ndarray or list of numpy.ndarray): an array or list of arrays containing training data.
            y (numpy.ndarray or list of numpy.ndarray): an array or list of arrays containing label data.
            *args (list): Ordered arguments.
            **kwargs (dict): Keyword arguments.

        Returns:
            dict: Output of the validation function.
        """

        return {}

    @abstractmethod
    def predict(self, x, *args, **kwargs):
        """
        Use the network for evaluation.

        Args:
            x (numpy.ndarray or list of numpy.ndarray): an array or list of arrays containing training data.
            *args (list): Ordered arguments.
            **kwargs (dict): Keyword arguments.

        Returns:
            dict: Output of the evaluation function.
        """

        return {}

    @abstractmethod
    def updatelearningrate(self, learning_rate):
        """
        Update the learnig rate.

        Args:
            learning_rate (float): New learning rate.
        """

        pass

    @abstractmethod
    def getnumberofoutputchannels(self):
        """
        Returns the output channels of the network
        """

        pass

    @abstractmethod
    def metricnames(self):
        """
        Get the list of metric names that the network returns.
        Returns:
            list: Metric names.
        """

        pass

    @abstractmethod
    def getreconstructioninformation(self, input_shape=None):
        """
        Calculate the scale factor and padding to reconstruct an input shape.

        This function determines which layers are used to calculate the reconstruction information for a network. If
        input_shape is not give it will be extracted from the network.

        Args:
            input_shape (tuple of ints): The input shape for which to calculate the reconstruction information.

        Returns:
            np.array: lost pixels
            np.array: downsample factor

        Raises:
            MissingNetworkError: The network is not defined.
        """

        return [], []

    @abstractmethod
    def _getreconstructioninformationforlayers(self, input_shape, layers):
        """
        Calculate the scale factor and padding to reconstruct the input shape for a given set of layers.

        This function calculates the information needed to reconstruct the input image shape given a set of layers.

        Args:
            input_shape (tuple of ints): The input shape for which to calculate the reconstruction information.
            layers (list of backend layer objects): selected list of connected layers for which the reconstruction
                information needs to be calculated.

        Returns:
            np.array: lost pixels
            np.array: downsample factor

        Raises:
            MissingNetworkError: The network is not defined.
        """

        return [], []
