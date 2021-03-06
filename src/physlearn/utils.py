from datetime import datetime, timedelta
from hashlib import md5
from logging import getLogger
from pathlib import Path
from time import time
from typing import Union

import _pickle as pickle
import h5py
import numpy as np

from physlearn.names import HospitalName

SIGNATURE_FILE_SUFFIX = ".sign"
SIGN_DELIMITER = ":"


_LOG = getLogger()


def create_time_axis(start: datetime, length: int, dt: timedelta) -> np.ndarray:
    """
    Creates a time axis of a given length from a given start time. Time interval
    between adjacent values is determined by dt.

    Args:
      start: datetime object indicating start of time axis
      length: int: length of time axis created
      dt: timedelta: time difference between adjacent values in time axis,
      corresponds to the sampling frequency.

    Returns:
      time axis: with given length, starting from start time.
      Time difference between consecutive samples is determined by dt.

    """
    end = start + dt * length
    time_axis = np.arange(start, end, dt).astype(datetime)
    return time_axis


def find_files_with_extension(extension: str, files_path: Path):
    """
    Finds files recursively from the current path with the given extension.

    Args:
      extension: string of desired file extension ('edf', 'txt', etc)
      files_path: path for txt files

    Returns:
      files: a list of file paths (str) with the extension

    """
    return list(Path(files_path).glob("**/*." + extension))


def check_hdf5_file_validity(path):
    """
    Check that the given path points to an hdf5 file.

    Raises:
        Value error if the file is not an hdf5 file.
    """
    if not h5py.is_hdf5(path):
        raise ValueError("The given path is not an hdf5 file")


def find_digits_in_str(string: str) -> str:
    """Find digits in a given string.

    Args:
      string: str, input string with the desired digits

    Returns:
      digits: str, found in the given string

    """
    return "".join(x for x in string if x.isdigit())


def assert_hospital(hospital_in: HospitalName, hospital_check: HospitalName):
    """Assert hospital_in matches the hospital_check.
    Functions build for a specific hospital check match before continuing in order to
    prevent errors later on.

    Args:
      hospital_in: HospitalType of the input
      hospital_check: HospitalType intended

    Returns:
      : none

    """
    msg = f"Hospital mismatch: {hospital_in} and {hospital_check.name}"
    if hospital_in.name != hospital_check.name:
        raise ValueError(msg)


def bytes2str(value: Union[str, bytes]) -> str:
    """
    Converts bytes argument to a string if needed

    Args:
        value: either a string or a bytes argument

    Returns: a string argument
    """
    if isinstance(value, str):
        return value
    return value.decode("utf-8", errors="ignore")


def safe_pickle(directory: str, file_name: str, obj):
    """Safely pickle an object

    Never again corrupt important pickles again, saves an object to a pickle  with a
    unique name.


    Args:
        directory: The directory to save to as a string
        file_name: The file name (without the extension) to use as a string
        obj: The object to pickle

    """
    # Adding time signature to make the file name unique
    time_signature = str(hex(int(time() * 1000)))[2:]

    path = directory + file_name + f"{time_signature}.pkl"

    if Path(path).exists():
        raise RuntimeError(
            "Could not create a unique name, please try again, should "
            "probably work then"
        )

    # Save the file
    with open(directory + file_name + f"{time_signature}.pkl", "wb") as f:
        pickle.dump(obj, f)


def find_letters_in_str(string: str) -> str:
    """Find letters in a given string.

    Args:
      string: str, input string

    Returns:
      letters: str, found in the given string

    """
    letters = "".join(x for x in string if x.isalpha())
    return letters


def cached_sign(path: Path) -> str:
    """Signs files using signature cache

    Tries to load the signature from a signature cache. If signature cache exists
    and the OS time of last modification of the file didn't change, returns the cached
    signature, if fails, calculates the signature, caches it and returns it.

    The cache is a simple text file with the same name and extension of ".sign"

    Args:
        path: path of the file to sign

    Returns: File md5 signature as a string

    """

    sign_file = path.with_suffix(SIGNATURE_FILE_SUFFIX)
    last_modified = str(path.stat().st_mtime_ns)
    if sign_file.is_file():
        with sign_file.open("rt") as sf:
            last_modified_cache, sign = sf.read().split(SIGN_DELIMITER)
            if last_modified == last_modified_cache:
                return sign
            else:
                _LOG.info("File changed since last signing, recalculating signature.")

    with path.open(mode="rb") as f:
        _LOG.info(f"calculating signature for file {path}")
        file_hash = md5()
        chunk = f.read(8192)
        while chunk:
            file_hash.update(chunk)
            chunk = f.read(8192)
    sign = file_hash.hexdigest()

    with sign_file.open("wt") as sf:
        sf.write(last_modified + SIGN_DELIMITER + sign)
    sign_file.chmod(0o777)

    return sign
