# -*- coding: utf-8 -*-
# Copyright (C) 2020 Unbabel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import subprocess
import urllib.request
import zipfile
from typing import List
from urllib.parse import urlparse

from tqdm import tqdm

from comet.models import available_metrics

logger = logging.getLogger(__name__)


def get_cache_folder():
    if "HOME" in os.environ:
        cache_directory = os.environ["HOME"] + "/.cache/torch/unbabel_comet/"
        if not os.path.exists(cache_directory):
            os.makedirs(cache_directory)
        return cache_directory
    else:
        raise Exception("HOME environment variable is not defined.")


def _reporthook(t):
    """``reporthook`` to use with ``urllib.request`` that prints the
        process of the download.

    Uses ``tqdm`` for progress bar.

    **Reference:**
    https://github.com/tqdm/tqdm

    """
    last_b = [0]

    def inner(b: int = 1, bsize: int = 1, tsize: int = None):
        """
        :param b: Number of blocks just transferred [default: 1].
        :param bsize: Size of each block (in tqdm units) [default: 1].
        :param tsize: Total size (in tqdm units).
            If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner


def _maybe_extract(compressed_filename: str, directory: str, extension: str = None):
    """Extract a compressed file to ``directory``.

    :param compressed_filename: Compressed file.
    :param directory: Extract to directory.
    :param extension: Extension of the file; Otherwise, attempts to
        extract extension from the filename.
    """
    logger.info("Extracting {}".format(compressed_filename))

    if extension is None:
        basename = os.path.basename(compressed_filename)
        extension = basename.split(".", 1)[1]

    if "zip" in extension:
        with zipfile.ZipFile(compressed_filename, "r") as zip_:
            zip_.extractall(directory)
    elif "tar.gz" in extension or "tgz" in extension:
        # `tar` is much faster than python's `tarfile` implementation
        subprocess.call(["tar", "-C", directory, "-zxvf", compressed_filename])
    elif "tar" in extension:
        subprocess.call(["tar", "-C", directory, "-xvf", compressed_filename])

    logger.info("Extracted {}".format(compressed_filename))


def _get_filename_from_url(url):
    """Return a filename from a URL

    Args:
        url (str): URL to extract filename from

    Returns:
        (str): Filename in URL
    """
    parse = urlparse(url)
    return os.path.basename(parse.path)


def _check_download(*filepaths):
    """Check if the downloaded files are found.

    Args:
        filepaths (list of str): Check if these filepaths exist

    Returns:
        (bool): Returns True if all filepaths exist
    """
    return all([os.path.isfile(filepath) for filepath in filepaths])


def download_file_maybe_extract(
    url: str,
    directory: str,
    filename: str = None,
    extension: str = None,
    check_files: List[str] = [],
):
    """Download the file at ``url`` to ``directory``.
        Extract to ``directory`` if tar or zip.

    :param url: Url of file (str or Path).
    :param directory: Directory to download to.
    :param filename: Name of the file to download; Otherwise, a filename is extracted
        from the url.
    :param extension: Extension of the file; Otherwise, attempts to extract extension
        from the filename.
    :param check_files: Check if these files exist, ensuring the download
        succeeded. If these files exist before the download, the download is skipped.

    :return: Filename of download file.
    """
    if filename is None:
        filename = _get_filename_from_url(url)

    directory = str(directory)
    filepath = os.path.join(directory, filename)
    check_files = [os.path.join(directory, str(f)) for f in check_files]

    if len(check_files) > 0 and _check_download(*check_files):
        return filepath

    if not os.path.isdir(directory):
        os.makedirs(directory)

    logger.info("Downloading {}".format(filename))

    # Download
    with tqdm(unit="B", unit_scale=True, miniters=1, desc=filename) as t:
        urllib.request.urlretrieve(url, filename=filepath, reporthook=_reporthook(t))

    _maybe_extract(
        compressed_filename=filepath, directory=directory, extension=extension
    )

    if not _check_download(*check_files):
        raise ValueError("[DOWNLOAD FAILED] `*check_files` not found")

    return filepath


def download_model(model: str, saving_directory: str = None) -> str:
    """
    Function that loads pretrained models from AWS.

    :param model: Name of the model to be loaded.
    :param saving_directory: RELATIVE path to the saving folder (must end with /).

    Return:
        - Path to model checkpoint.
    """

    if saving_directory is None:
        saving_directory = get_cache_folder()

    if not saving_directory.endswith("/"):
        saving_directory += "/"

    if not os.path.exists(saving_directory):
        os.makedirs(saving_directory)

    if os.path.isdir(saving_directory + model):
        logger.info(f"{model} is already in cache.")
        if not model.endswith("/"):
            model += "/"

    elif model not in available_metrics.keys():
        raise Exception(
            f"{model} is not in the `availale_metrics` or is a valid checkpoint folder."
        )

    elif available_metrics[model].startswith("https://"):
        download_file_maybe_extract(
            available_metrics[model], directory=saving_directory
        )

    else:
        raise Exception("Invalid model name!")

    # CLEAN Cache
    if os.path.exists(saving_directory + model + ".zip"):
        os.remove(saving_directory + model + ".zip")
    if os.path.exists(saving_directory + model + ".tar.gz"):
        os.remove(saving_directory + model + ".tar.gz")
    if os.path.exists(saving_directory + model + ".tar"):
        os.remove(saving_directory + model + ".tar")

    checkpoints_folder = saving_directory + model + "/checkpoints"
    checkpoints = [
        file for file in os.listdir(checkpoints_folder) if file.endswith(".ckpt")
    ]
    checkpoint = checkpoints[-1]
    checkpoint_path = checkpoints_folder + "/" + checkpoint
    return checkpoint_path
