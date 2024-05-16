"""
This file contains a function that can be executed in a separate process or thread for extracting patches from a collection of whole slide images.
"""

from . import batchsampler as dptbatchsampler

from ..errors import processerrors as dptprocesserrors
from ..utils import trace as dpttrace

import queue
import threading
import sys

#----------------------------------------------------------------------------------------------------

def batchsampler_daemon_loop(command_queue,
                             response_queue,
                             label_dist,
                             patch_shapes,
                             mask_spacing,
                             spacing_tolerance,
                             input_channels,
                             label_mode,
                             patch_sources,
                             category_dist,
                             strict_selection,
                             create_stats,
                             patch_augmenter,
                             label_mapper,
                             free_label_range,
                             weight_mapper,
                             process_count,
                             pool_size,
                             join_timeout,
                             response_timeout,
                             poll_timeout,
                             name_tag):
    """
    Create a message processing loop that can be executed in a separate thread. The loop can create a batch sampler and extract patches from an image. For
    performance considerations on the 'sample' command the loop inserts the sampled data directly to the patch buffer instead of sending it in a response
    message.

    Args:
        command_queue (queue.Queue): Command queue for thread communication. Dictionary messages accepted.
            The message is identified by the 'command' key. Accepted command values:
                'init': Initialize the sampler.
                'batch': Sample a batch and push it to the patch buffer.. Further key expected: 'count'.
                'step': Switch to the next collection of patch samplers.
                'stop': Quit processing loop and join all sampler processes
            Further keys that are accepted:
                'count': Number of patches to extract with 'batch' command.
                'buffer': Buffer object where the results shall be put.
        response_queue (queue.Queue): Queue where the occurred errors are sent. The response key will be 'error'.
            Further keys in the error messages:
                'tid': The thread identifier of this thread. It is attached to every response message.
                'exception': Exception object sent with 'error' messages.
                'trace': Traceback information for the exception.
        label_dist (dict): Label sampling distribution mapping from label value to ratio in a single batch.
        patch_shapes (dict): Desired patch shapes (rows, cols) per level.
        mask_spacing (float): Pixel spacing of the masks to process (micrometer).
        spacing_tolerance (float): Pixel spacing tolerance (percentage).
        input_channels (list) : Desired channels that are extracted for each patch.
        label_mode (str): Label generation mode. Accepted values:
            'central': Just return the label of the central pixel.
            'synthesize': Synthesize the label map from the mask statistics and zoom it to the appropriate level.
            'load': Load the label map from the label image and zoom it to the appropriate level if necessary.
        patch_sources (dict): Data source: map from image categories to PatchSource object sets.
        category_dist (dict): Image category sampling distribution mapping from image category to ratio in a single batch.
        strict_selection (bool): If true, every label that has higher than 0.0 ratio must be available in every source image selection.
        create_stats (bool): Allow missing stat files, and create them if necessary.
        patch_augmenter (augmenters.AugmenterPool, None): Configured patch augmenter object.
        label_mapper (label.LabelMapper): Label value to label index mapper object.
        free_label_range (bool): If this flag is True, non-continuous label ranges, and ranges that do not start at 0 are also allowed.
        weight_mapper (weight.WeightMapperBase, None): Label weight map calculator object.
        process_count (int): Number of processes to use during extraction at once. 0 means that there will be no external worker processes.
        pool_size (int): Number of images to open or worker processes to spawn at once for extraction.
        join_timeout (int): Seconds to wait for child processes to join.
        response_timeout (int): Seconds to wait for inter-process communication responses. 0 means no timeout.
        poll_timeout (int): Seconds to wait for a message. Safeguard against hanging.
        name_tag (str): Name tag of the sampler for logging.
    """

    # Thread ID for identifying the sender.
    #
    tid = threading.current_thread().ident

    # Enter processing loop.
    #
    batch_sampler = None
    command_message = None
    execute_loop = True

    while execute_loop:
        # Exceptions could be raised during sampler object instantiation.
        #
        try:
            # Extract command from the connection. Try to dissect if it is a tuple.
            #
            command_message = command_queue.get(timeout=poll_timeout)

            # Execute the command.
            #
            if command_message['command'] == 'init':
                # Create sampler object that can extract patches from a single image.
                #
                batch_sampler = dptbatchsampler.BatchSampler(label_dist=label_dist,
                                                             patch_shapes=patch_shapes,
                                                             mask_spacing=mask_spacing,
                                                             spacing_tolerance=spacing_tolerance,
                                                             input_channels=input_channels,
                                                             label_mode=label_mode,
                                                             patch_sources=patch_sources,
                                                             category_dist=category_dist,
                                                             strict_selection=strict_selection,
                                                             create_stats=create_stats,
                                                             patch_augmenter=patch_augmenter,
                                                             label_mapper=label_mapper,
                                                             free_label_range=free_label_range,
                                                             weight_mapper=weight_mapper,
                                                             process_count=process_count,
                                                             pool_size=pool_size,
                                                             join_timeout=join_timeout,
                                                             response_timeout=response_timeout,
                                                             poll_timeout=poll_timeout,
                                                             name_tag=name_tag)

            elif command_message['command'] == 'batch':
                # Extract and send patches.
                #
                patch_dict = batch_sampler.batch(batch_size=command_message['count'])
                command_message['buffer'].push(patches=patch_dict)

            elif command_message['command'] == 'step':
                # Switch patch samplers.
                #
                batch_sampler.step()

            elif command_message['command'] == 'stop':
                # Terminate the loop.
                #
                execute_loop = False

            elif command_message['command'] == 'ping':
                # Check all worker processes.
                #
                if batch_sampler:
                    batch_sampler.ping()

                # Signal if the thread is ready.
                #
                response_queue.put({'response': 'pong', 'tid': tid, 'pids': batch_sampler.workers})

            else:
                # Invalid message: send the exception and terminate the thread. Exceptions cannot be rebuilt from serialized versions. So sent string in the message.
                #
                response_queue.put({'response': 'error', 'exception': repr(dptprocesserrors.InvalidMessageError(command_message)), 'trace': '', 'command': command_message, 'tid': tid})
                execute_loop = False

        except queue.Empty as empty_queue:
            # Collect and summarize traceback information.
            #
            _, _, exception_traceback = sys.exc_info()
            trace_string = dpttrace.format_traceback(traceback_object=exception_traceback)

            # Log the event by the batch sampler too.
            #
            if batch_sampler:
                batch_sampler.error(message='Poll timeout: no message received for {timeout} secs. Shutdown is imminent'.format(timeout=poll_timeout))

            # There was no message received from the queue before timeout. Exceptions cannot be rebuilt from serialized versions. So sent string in the message.
            #
            response_queue.put({'response': 'error', 'exception': repr(empty_queue), 'trace': trace_string, 'tid': tid})
            execute_loop = False

        except Exception as exception:
            # Collect and summarize traceback information.
            #
            _, _, exception_traceback = sys.exc_info()
            trace_string = dpttrace.format_traceback(traceback_object=exception_traceback)

            # Log the event by the batch sampler too.
            #
            if batch_sampler:
                batch_sampler.error(message='Exception raised: \'{exception}\'. Trace: \'{trace}\'. Shutdown is imminent'.format(exception=repr(exception), trace=trace_string))

            # Send the error back to the server master process in case of error and terminate. Exceptions cannot be rebuilt from serialized versions. So sent string in the message.
            #
            response_queue.put({'response': 'error', 'exception': repr(exception), 'trace': trace_string, 'command': command_message, 'tid': tid})
            execute_loop = False

    # Shut down the worker processes.
    #
    if batch_sampler:
        batch_sampler.stop()
