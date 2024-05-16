"""
This module can retrieve basic information from a Git repository.
"""

import git
import re

#----------------------------------------------------------------------------------------------------

def git_info(repo_dir_path):
    """
    Get basic information about a Git repository.

    Args:
        repo_dir_path (str): Directory path of a repository.

    Returns:
        dict, None: A dictionary of the retrieved information with 'name', 'url', 'revision' and 'branch' keys or None in case of error.
    """

    # Build a repository object and obtain basic information: name, URL, revision, and branch.
    #
    try:
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
