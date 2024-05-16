"""
This file contains a function that can be executed in a separate process or thread for saving experiment statistics.
"""

from . import statshandler as dptstatshandler

from ..errors import processerrors as dptprocesserrors
from ..utils import trace as dpttrace

import shutil
import os
import sys

#----------------------------------------------------------------------------------------------------

def statshandler_daemon_loop(command_pipe, epoch_save_path, epoch_plot_path, epoch_stats_to_plot, experiment_name, append, poll_timeout):
    """
    Create a message processing loop that can be executed in a separate thread or process. The loop can create a stats handler that saves the stats and copy files.

    Args:
        command_pipe (multiprocessing.Connection): Child end of a pipe of inter-process communication. Dictionary messages accepted.

        epoch_save_path (str, None): Epoch statistics save file path.
        epoch_plot_path (str, None): Epoch statistics plot file path.
        epoch_stats_to_plot (list): List of statistics to plot.
        experiment_name (str): Experiment name for plotting.
        append (bool): Whether to overwrite existing statistics files instead of appending.
        poll_timeout (int): Seconds to wait for a message. Safeguard against hanging.
    """

    # Process ID for identifying the sender.
    #
    pid = os.getpid()

    # Enter processing loop.
    #
    stats_handler = None  # Stats handler object.
    execute_loop = True   # Flag to terminate loop function.
    exit_code = -1        # Exit code to return error codes.
    write_attempts = 5    # Number of write attempts for file copying.

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
                if command_message['command'] == 'init':
                    # Create stats handler object that can save stats.
                    #
                    stats_handler = dptstatshandler.StatsHandler(epoch_save_path=epoch_save_path,
                                                                 epoch_plot_path=epoch_plot_path,
                                                                 epoch_stats_to_plot=epoch_stats_to_plot,
                                                                 experiment_name=experiment_name,
                                                                 append=append)

                elif command_message['command'] == 'stats':
                    # Add epoch statistics.
                    #
                    stats_handler.append(epoch_statistics_row=command_message['epoch'])
                    stats_handler.save()
                    stats_handler.plot()

                elif command_message['command'] == 'copy':
                    # Copy files.
                    #
                    for attempt in range(write_attempts):
                        try:
                            shutil.copyfile(src=command_message['source'], dst=command_message['destination'])
                            break

                        except OSError as exception:
                            # Copy attempt failed. Send back details for logging.
                            #
                            command_pipe.send({'response': 'warning',
                                               'reason': 'copy',
                                               'source': command_message['source'],
                                               'destination': command_message['destination'],
                                               'attempt': attempt,
                                               'total': write_attempts,
                                               'exception': repr(exception),
                                               'trace': '',
                                               'pid': pid})

                elif command_message['command'] == 'rewind':
                    # Rewind to the given epoch.
                    #
                    stats_handler.rewind(index=command_message['index'])

                elif command_message['command'] == 'stop':
                    # Terminate the process.
                    #
                    execute_loop = False
                    exit_code = 0

                elif command_message['command'] == 'ping':
                    # Test if the process is ready.
                    #
                    command_pipe.send({'response': 'pong', 'pid': pid})

                else:
                    # Invalid message: send the exception and terminate the process. Exceptions cannot be rebuilt from serialized versions. So send string in the message.
                    #
                    command_pipe.send({'response': 'error', 'exception': repr(dptprocesserrors.InvalidMessageError(command_message)), 'trace': '', 'pid': pid})
                    execute_loop = False
                    exit_code = 2

            else:
                # No message arrived before timeout. Shut down the process. Exceptions cannot be rebuilt from serialized versions. So send string in the message.
                #
                command_pipe.send({'response': 'error', 'exception': repr(dptprocesserrors.ProcessPollTimeoutError(poll_timeout)), 'trace': '', 'pid': pid})
                execute_loop = False
                exit_code = 3

        except Exception as exception:
            # Collect and summarize traceback information.
            #
            _, _, exception_traceback = sys.exc_info()
            trace_string = dpttrace.format_traceback(traceback_object=exception_traceback)

            # Send the error back to the master process in case of error and terminate. Exceptions cannot be rebuilt from serialized versions. So send string in the message.
            #
            command_pipe.send({'response': 'error', 'exception': repr(exception), 'trace': trace_string, 'pid': pid})
            execute_loop = False
            exit_code = 4

    # Close the connections.
    #
    command_pipe.close()

    # Exit process with the configured code.
    #
    sys.exit(exit_code)
