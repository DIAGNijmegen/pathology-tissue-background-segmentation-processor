"""
This module can collect the content of a folder with wildcards in the path.
"""

import glob
import os
import re

#----------------------------------------------------------------------------------------------------

def folder_content(folder_path, filter_exp=None, recursive=False):
    """
    Collect the content of a folder with wildcards in its name.

    Args:
        folder_path (str): Folder path to examine with wildcards in its name.
        filter_exp (str, None): Filter regular expression pattern to apply on the file base names.
        recursive (bool): Collect files recursively if a directory is given.

    Returns:
        list: List of collected file paths.
    """

    # Compile the regular expression if given.
    #
    filter_regex = re.compile(pattern=filter_exp) if filter_exp else None

    # Sort the initial wildcard replacement into files and folders.
    #
    static_list = glob.glob(folder_path)
    folders_to_traverse = []
    path_list = []
    for path_item in static_list:
        if os.path.isdir(path_item):
            folders_to_traverse.append(path_item)
        else:
            path_list.append(path_item)

    # Go through the list of directories to traverse.
    #
    folder_index = 0
    while folder_index < len(folders_to_traverse):
        items_in_folder = glob.glob(os.path.join(folders_to_traverse[folder_index], '*'))
        for path_item in items_in_folder:
            if os.path.isdir(path_item):
                # At this point all directories to traverse from the initial finding have been added. Add the found directories only if the recursive
                # flag have been set. Otherwise just the files in the initial directory findings are added.
                #
                if recursive:
                    folders_to_traverse.append(path_item)
            else:
                # Match the file base name against the filer regular expression, if given.
                #
                if filter_regex:
                    if filter_regex.match(os.path.basename(path_item)):
                        path_list.append(path_item)
                else:
                    path_list.append(path_item)

        folder_index += 1

    # Return the list of collected file paths.
    #
    return path_list
