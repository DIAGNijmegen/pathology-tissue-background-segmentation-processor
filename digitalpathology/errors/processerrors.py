"""
Multi-processing and multi-threading related errors.
"""

from . import errorbase as dpterrorbase

#----------------------------------------------------------------------------------------------------

class DigitalPathologyProcessError(dpterrorbase.DigitalPathologyError):
    """Error base class for all process errors."""

    def __init__(self, *args):
        """
        Initialize the object.

        Args:
            *args: Argument list.
        """

        # Initialize base class.
        #
        super().__init__(*args)

#----------------------------------------------------------------------------------------------------

class InvalidMessageError(DigitalPathologyProcessError):
    """Raise when the received message is invalid."""

    def __init__(self, message):
        """
        Initialize the object.

        Args:
            message: Message.
        """

        # Initialize base class.
        #
        super().__init__('Invalid message: <{message}>.'.format(message=message))

        # Store custom data.
        #
        self.message = message

#----------------------------------------------------------------------------------------------------

class InvalidPoolSizeError(DigitalPathologyProcessError):
    """Raise when the pool size is not valid."""

    def __init__(self, pool_size):
        """
        Initialize the object.

        Args:
            pool_size (int): Pool size.
        """

        # Initialize base class.
        #
        super().__init__('Invalid pool size: {size}.'.format(size=pool_size))

        # Store custom data.
        #
        self.pool_size = pool_size

#----------------------------------------------------------------------------------------------------

class ProcessTerminatedError(DigitalPathologyProcessError):
    """Raise when the process has terminated unexpectedly."""

    def __init__(self, process_id):
        """
        Initialize the object.

        Args:
            process_id (int): Process identifier.
        """

        # Initialize base class.
        #
        super().__init__('Process terminated unexpectedly: {pid}.'.format(pid=process_id))

        # Store custom data.
        #
        self.process_id = process_id

#----------------------------------------------------------------------------------------------------

class UnexpectedProcessResponseError(DigitalPathologyProcessError):
    """Raise when the received response is unexpected or invalid."""

    def __init__(self, process_pid, message, response):
        """
        Initialize the object.

        Args:
            process_pid (int): Process identifier.
            message (str, dict): Sent message.
            response: Received response.
        """

        # Initialize base class.
        #
        super().__init__('Invalid message exchange: pid={pid}; message=<{message}>; response=<{response}>.'.format(pid=process_pid, message=message, response=response))

        # Store custom data.
        #
        self.process_pid = process_pid
        self.message = message
        self.response = response

#----------------------------------------------------------------------------------------------------

class ThreadTerminatedError(DigitalPathologyProcessError):
    """Raise when the thread has terminated unexpectedly."""

    def __init__(self, thread_id):
        """
        Initialize the object.

        Args:
            thread_id (int): Thread identifier.
        """

        # Initialize base class.
        #
        super().__init__('Thread terminated unexpectedly: {tid}.'.format(tid=thread_id))

        # Store custom data.
        #
        self.thread_id = thread_id

#----------------------------------------------------------------------------------------------------

class UnknownThreadResponseError(DigitalPathologyProcessError):
    """Raise when the received response is unknown."""

    def __init__(self, thread_tid, response):
        """
        Initialize the object.

        Args:
            thread_tid (int): Thread identifier.
            response: Received response.
        """

        # Initialize base class.
        #
        super().__init__('Unknown response from thread {tid}: <{response}>.'.format(tid=thread_tid, response=response))

        # Store custom data.
        #
        self.thread_tid = thread_tid
        self.response = response

#----------------------------------------------------------------------------------------------------

class ErrorThreadResponseError(DigitalPathologyProcessError):
    """Raise when the received response is an error message."""

    def __init__(self, thread_id, message, response):
        """
        Initialize the object.

        Args:
            thread_id (int): Thread identifier.
            message (str, dict): Sent message.
            response: Received response.
        """

        # Initialize base class.
        #
        super().__init__('Error message received: tid={tid}; message={message}; response={response}.'.format(tid=thread_id, message=message, response=response))

        # Store custom data.
        #
        self.thread_id = thread_id
        self.message = message
        self.response = response

#----------------------------------------------------------------------------------------------------

class InvalidTimeoutError(DigitalPathologyProcessError):
    """Raise when the response timeout value for inter-process communication is negative."""

    def __init__(self, timeout_purpose, timeout_secs):
        """
        Initialize the object.

        Args:
            timeout_purpose (str): Timeout purpose.
            timeout_secs (int): Seconds to wait for inter-process communication response.
        """

        # Initialize base class.
        #
        super().__init__('Invalid {purpose} timeout length: {secs}.'.format(purpose=timeout_purpose, secs=timeout_secs))

        # Store custom data.
        #
        self.timeout_purpose = timeout_purpose
        self.timeout_secs = timeout_secs

#----------------------------------------------------------------------------------------------------

class InvalidProcessCountError(DigitalPathologyProcessError):
    """Raise when the process count is negative."""

    def __init__(self, process_count):
        """
        Initialize the object.

        Args:
            process_count (int): Number of worker processes to spawn.
        """

        # Initialize base class.
        #
        super().__init__('Invalid process count: {count}.'.format(count=process_count))

        # Store custom data.
        #
        self.process_count = process_count

#----------------------------------------------------------------------------------------------------

class ProcessResponseTimeoutError(DigitalPathologyProcessError):
    """Raise when the no responses received in time from worker processes in time."""

    def __init__(self, timeout_secs, queue_error):
        """
        Initialize the object.

        Args:
            timeout_secs (int): Timeout seconds.
            queue_error (Queue.Empty): Captured Queue error.
        """

        # Initialize base class.
        #
        super().__init__('No message received before timeout: {secs} secs.'.format(secs=timeout_secs))

        # Store custom data.
        #
        self.timeout_secs = timeout_secs
        self.queue_error = queue_error

#----------------------------------------------------------------------------------------------------

class ProcessPollTimeoutError(DigitalPathologyProcessError):
    """Raise when there is no request messages received in worker processes before the timeout."""

    def __init__(self, timeout_secs):
        """
        Initialize the object.

        Args:
            timeout_secs (int): Timeout seconds.
\        """

        # Initialize base class.
        #
        super().__init__('No request received before timeout: {secs} secs.'.format(secs=timeout_secs))

        # Store custom data.
        #
        self.timeout_secs = timeout_secs

#----------------------------------------------------------------------------------------------------

class ThreadJobTimeoutError(DigitalPathologyProcessError):
    """Raise when the thread did not finished its jobs in time."""

    def __init__(self, timeout_secs, thread_tid, queue_error):
        """
        Initialize the object.

        Args:
            thread_tid (int): Thread identifier.
            timeout_secs (float): Timeout seconds.
            queue_error (Queue.Empty): Captured Queue error.
        """

        # Initialize base class.
        #
        super().__init__('No message received from thread {tid} before timeout: {secs} secs.'.format(tid=thread_tid, secs=timeout_secs))

        # Store custom data.
        #
        self.thread_tid = thread_tid
        self.timeout_secs = timeout_secs
        self.queue_error = queue_error

#----------------------------------------------------------------------------------------------------

class InvalidSamplerChunkSizeError(DigitalPathologyProcessError):
    """Raise when the sampler chunk size is invalid."""

    def __init__(self, chunk_size):
        """
        Initialize the object.

        Args:
            chunk_size (int): Number of patches to read at once.
        """

        # Initialize base class.
        #
        super().__init__('Invalid sampler chunk size: {size}.'.format(size=chunk_size))

        # Store custom data.
        #
        self.chunk_size = chunk_size
