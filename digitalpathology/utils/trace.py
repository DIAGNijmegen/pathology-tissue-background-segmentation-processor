"""
This module can format traceback objects to strings for logging.
"""

import traceback
import os

#----------------------------------------------------------------------------------------------------

def format_traceback(traceback_object):
    """
    Format the system traceback to a string with file:line:function entries for logging.

    Args:
        traceback_object (traceback.traceback): Traceback object.

    Returns:
        str: formatted string of the traceback.
    """

    trace_summary = traceback.extract_tb(traceback_object)
    trace_info = ['{file}:{line}:{name}'.format(file=os.path.basename(summary_item.filename), line=summary_item.lineno, name=summary_item.name) for summary_item in trace_summary]

    return '/'.join(trace_info)
