"""
This file contains a function that can be executed in a separate process or thread for sampling patches from whole slide images.
"""

from . import patchsampler as dptpatchsampler

from ..errors import processerrors as dptprocesserrors
from ..utils import trace as dpttrace

import numpy as np
import random
import os
import sys

#----------------------------------------------------------------------------------------------------

def patchsampler_daemon_loop(command_pipe, response_queue, poll_timeout):
    """
    Create a message processing loop that can be executed in a separate thread or process. The loop can create a patch sampler and extract patches from an image.

    Args:
        command_pipe (multiprocessing.Connection): Child end of a pipe of inter-process communication. Dictionary messages accepted.
            The message is identified by the 'command' key. Accepted command values:
                'create': Initialize a sampler for the image. Further keys expected: 'source'.
                'remove': Remove the sampler for the image. Further keys expected: 'source'.
                'random': Initialize random number generators.
                'hooks': Configure the patch augmenter pool and label mapper objects. Further keys expected: 'augmenter' and 'mapper'.
                'sample': Sample patches. Further keys expected: 'count' and 'shape'.
                'stop': Quit processing loop.
                'ping': Request a 'ready' response.
            Further keys that are accepted:
                'source': PatchSource for 'create', 'remove' and 'sample' messages.
                'stat': Allow missing stat files, and create them if necessary.
                'spacing': Mask pixel spacing to use.
                'seed': Random number generator seed.
                'augmenter': Patch augmenter object for 'hooks' command.
                'labeler': Label mapper object for 'hooks' command.
                'weigher': Weight mapper object for 'hooks' command.
                'tolerance': Pixel spacing tolerance with the 'create' command.
                'channels': Channel list to configure with the 'create' command.
                'counts': List of number of patches to extract per label index.
                'shapes': Dictionary mapping level to (rows, columns) patch shape with 'sample' command.
                'mode': Label generation mode: 'central', 'synthesize' or 'load'.
        response_queue (multiprocessing.Queue): Queue where the responses are sent.
            Sent responses:
                'done': Answer for the 'create' or 'remove' commands.
                'pong: Answer for the 'ping' command.
                'error': Error occurred.
                'data': The response contains the requested data.
            Further keys in the messages:
                'pid': The PID of this process. It is attached to every response message.
                'exception': Exception object sent with 'error' messages.
                'trace': Traceback information for the exception.
                'patches': numpy.ndarray of patches attached to 'data' messages.
                'labels': numpy.ndarray of labels attached to 'data' messages.
        poll_timeout (int): Seconds to wait for a message. Safeguard against hanging.
    """

    # Process ID for identifying the sender.
    #
    pid = os.getpid()

    # Enter processing loop.
    #
    patch_samplers = {}     # Patch sampler dictionary, items identified by their source.
    patch_augmenter = None  # Patch augmenter pool.
    label_mapper = None     # Label mapper.
    weight_mapper = None    # Weight mapper.
    execute_loop = True     # Flag to terminate loop function.
    exit_code = -1          # Exit code to return error codes.

    while execute_loop:
        # Exceptions could be raised during sampler object instantiation.
        #
        try:
            # Check if there is message to read.
            #
            if command_pipe.poll(poll_timeout):
                # Extract command from the connection. Try to dissect if it is a tuple.
                #
                command_message = command_pipe.recv()

                # Execute the command.
                #
                if command_message['command'] == 'create':
                    # Create sampler object that can extract patches from a single image.
                    #
                    patch_samplers[command_message['source']] = dptpatchsampler.PatchSampler(patch_source=command_message['source'],
                                                                                             create_stat=command_message['stat'],
                                                                                             mask_spacing=command_message['spacing'],
                                                                                             spacing_tolerance=command_message['tolerance'],
                                                                                             input_channels=command_message['channels'],
                                                                                             label_mode=command_message['mode'])

                    response_queue.put({'response': 'done', 'pid': pid})

                elif command_message['command'] == 'remove':
                    # Remove a patch sampler object.
                    #
                    del patch_samplers[command_message['source']]

                    response_queue.put({'response': 'done', 'pid': pid})

                elif command_message['command'] == 'random':
                    # Initialize the random number generators. If the seed is None, the system time is used as seed.
                    #
                    random.seed(a=command_message['seed'], version=2)
                    np.random.seed(seed=random.randint(a=0, b=np.iinfo(np.uint32).max))

                elif command_message['command'] == 'hooks':
                    # Set the patch augmenter pool.
                    #
                    patch_augmenter = command_message['augmenter']
                    label_mapper = command_message['labeler']
                    weight_mapper = command_message['weigher']

                    response_queue.put({'response': 'done', 'pid': pid})

                elif command_message['command'] == 'sample':
                    # Extract and send patches from the selected sources and apply augmentations, label mapping, and weight mapping.
                    #
                    extract_shapes = patch_augmenter.shapes(target_shapes=command_message['shapes']) if patch_augmenter is not None else command_message['shapes']
                    patches = patch_samplers[command_message['source']].sample(counts=command_message['counts'], shapes=extract_shapes)

                    if patch_augmenter is not None:
                        patches = patch_augmenter.process(patches=patches, shapes=command_message['shapes'], randomize=True)

                    if label_mapper is not None:
                        patches = label_mapper.process(patches=patches, valid_map=weight_mapper is not None)

                    if weight_mapper is not None:
                        patches = weight_mapper.calculate(patches=patches)

                    response_queue.put({'response': 'data', 'patches': patches, 'source': command_message['source'], 'pid': pid})

                elif command_message['command'] == 'stop':
                    # Terminate the process.
                    #
                    execute_loop = False
                    exit_code = 0

                elif command_message['command'] == 'ping':
                    # Test if the process is ready.
                    #
                    response_queue.put({'response': 'pong', 'pid': pid})

                else:
                    # Invalid message: send the exception and terminate the process. Exceptions cannot be rebuilt from serialized versions. So send string in the message.
                    #
                    response_queue.put({'response': 'error', 'exception': repr(dptprocesserrors.InvalidMessageError(command_message)), 'trace': '', 'pid': pid})
                    execute_loop = False
                    exit_code = 1

            else:
                # No message arrived before timeout. Shut down the process. Exceptions cannot be rebuilt from serialized versions. So send string in the message.
                #
                response_queue.put({'response': 'error', 'exception': repr(dptprocesserrors.ProcessPollTimeoutError(poll_timeout)), 'trace': '', 'pid': pid})
                execute_loop = False
                exit_code = 2

        except Exception as exception:
            # Collect and summarize traceback information.
            #
            _, _, exception_traceback = sys.exc_info()
            trace_string = dpttrace.format_traceback(traceback_object=exception_traceback)

            # Send the error back to the master process in case of error and terminate. Exceptions cannot be rebuilt from serialized versions. So send string in the message.
            #
            response_queue.put({'response': 'error', 'exception': repr(exception), 'trace': trace_string, 'pid': pid})
            execute_loop = False
            exit_code = 3

    # Close the connections.
    #
    command_pipe.close()
    response_queue.close()

    # Prevent the other process block on join if there are unfinished tasks in the Queue that can happen in case of errors, when the master process shuts down all
    # sampler processes while some of them might still sampling. In this case when the sampler process receives the join and terminate command it already put a
    # sampled patch collection to the Queue that the master process will not consume.
    #
    response_queue.cancel_join_thread()

    # Exit process with the configured code.
    #
    sys.exit(exit_code)
