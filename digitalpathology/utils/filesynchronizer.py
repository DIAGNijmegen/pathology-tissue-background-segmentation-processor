"""
This file contains a class for synchronizing output files.
"""

from ..errors import trainingerrors as dpttrainingerrors

import os
import shutil

#----------------------------------------------------------------------------------------------------

class FileSynchronizer(object):
    """This class can synchronizer a collection of files between locations."""

    def __init__(self, work_directory):
        """
        Initialize the object. Save the work directory.

        Args:
            work_directory (str, None): Work directory.
        """

        # Initialize the base class.
        #
        super().__init__()

        # Initialize members.
        #
        self.__work_directory = work_directory  # Work directory.
        self.__sync_items = {}                  # Target to work path map.
        self.__copy_attempts = 4                # Number of copy attempts.

    def __relocatefile(self, work_path, target_path, move_file):
        """
        Copy a file from the work to the target path.

        Args:
            work_path (str): Source file path.
            target_path (str): Target file path.
            move_file (bool): Move file to the target path instead of copying.

        Returns:
            copied (bool): True if the file is successfully copied, False otherwise.
        """

        # Attempt to move or copy the file.
        #
        attempt = 0
        copied = False
        while attempt <= self.__copy_attempts and not copied:
            attempt += 1

            try:
                if move_file:
                    shutil.move(src=work_path, dst=target_path, copy_function=shutil.copyfile)
                else:
                    shutil.copyfile(src=work_path, dst=target_path)

                copied = True

            except shutil.SameFileError:
                break

            except FileNotFoundError:
                break

            except OSError:
                pass

        return copied

    def add(self, target_path):
        """
        Add file path to the synchronization list.

        Args:
            target_path (str): Target path.

        Raises:
            InvalidSyncTargetPathError: Invalid sync target path.
        """

        # Check if the target path is valid.
        #
        if target_path is None:
            raise dpttrainingerrors.InvalidSyncTargetPathError(target_path)

        # Check if work directory is valid. Otherwise synchronization is disabled.
        #
        if self.__work_directory is not None:
            # Calculate the work path.
            #
            work_path = os.path.join(self.__work_directory, os.path.basename(target_path))

            # Check if the local path is already in use. Append index if necessary.
            #
            if work_path in self.__sync_items.values():
                work_path_parts = os.path.splitext(work_path)

                index = 0
                present = True
                while present:
                    work_path = '{part[0]}_{suffix}{part[1]}'.format(part=work_path_parts, suffix=index)

                    if work_path in self.__sync_items.values():
                        index += 1
                    else:
                        present = False

            # Add calculated unique path to the sync path map.
            #
            self.__sync_items[target_path] = work_path

    def work(self, target_path):
        """
        Return the work path for the target path.

        Args:
            target_path (str, None): Target path.

        Returns:
            (str): Work path for the target path,
        """

        return self.__sync_items.get(target_path, target_path)

    def sync(self, target_path=None, move=False):
        """
        Synchronize files from the work path to their target path.

        Args:
            target_path (str, list, None): Target path, or list of target paths to sync.
            move (bool): Move files instead of copying.

        Raises:
            FileSyncFailedError: The synchronization of some files failed.
        """

        # Collect the unsuccessful copies.
        #
        unsuccessful_targets = []

        # Collect items to sync.
        #
        if target_path is None:
            sync_items = self.__sync_items.keys()
        else:
            sync_items = [target_path] if type(target_path) == str else list(target_path)

        # Copy all items.
        #
        for target_path_item in sync_items:
            work_path = self.__sync_items.get(target_path_item, None)
            if work_path is not None:
                if not self.__relocatefile(work_path=work_path, target_path=target_path_item, move_file=move):
                    unsuccessful_targets.append(target_path_item)
            else:
                unsuccessful_targets.append(work_path)

        # Raise exception in case of unsuccessful copies.
        #
        if unsuccessful_targets:
            dpttrainingerrors.FileSyncFailedError(unsuccessful_targets)

    def back(self, target_path=None, move=False):
        """
        Synchronize files back to their work path from their target path.

        Args:
            target_path (str, list, None): Target path, or list of target paths to sync.
            move (bool): Move files instead of copying.

        Raises:
            FileSyncFailedError: The synchronization of some files failed.
        """

        # Collect the unsuccessful copies.
        #
        unsuccessful_targets = []

        # Collect items to sync.
        #
        if target_path is None:
            sync_items = self.__sync_items.keys()
        else:
            sync_items = [target_path] if type(target_path) == str else list(target_path)

        # Copy all items.
        #
        for target_path_item in sync_items:
            work_path = self.__sync_items.get(target_path_item, None)
            if work_path is not None:
                if not self.__relocatefile(work_path=target_path_item, target_path=work_path, move_file=move):
                    unsuccessful_targets.append(self.__sync_items.get(target_path_item, target_path_item))
            else:
                unsuccessful_targets.append(work_path)

        # Raise exception in case of unsuccessful copies.
        #
        if unsuccessful_targets:
            dpttrainingerrors.FileSyncFailedError(unsuccessful_targets)
