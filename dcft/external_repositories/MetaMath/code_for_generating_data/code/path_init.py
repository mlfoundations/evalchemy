
from pathlib import Path

from utils.path_utils import PathUtils

"""
set the current working path
"""

work_path = Path().resolve().parent
PathUtils.HOME_PATH = work_path
PathUtils.set_path()
print(work_path)


def null_fun():
    pass