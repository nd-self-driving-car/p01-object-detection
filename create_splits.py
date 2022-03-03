import argparse
import glob
import os
import random

import numpy as np

from utils import get_module_logger


def split(source, destination):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, contains 3 sub folders: train / val / test
    """
    curr_dir = os.getcwd()
    if not os.path.isabs(source):
        source = os.path.join(curr_dir, source)
    if not os.path.isabs(destination):
        destination = os.path.join(curr_dir, destination)

    files = np.array(glob.glob(os.path.join(source, '*')))
    files_count = files.size
    train_indices = np.random.choice(files_count, int(files_count * 0.7), replace=False)
    train_files = (files[idx] for idx in train_indices)
    create_symlinks(train_files, os.path.join(destination, 'train'))

    files = np.delete(files, train_indices)
    files_count = files.size
    val_indices = np.random.choice(files_count, int(files_count * 0.66), replace=False)
    val_files = (files[idx] for idx in val_indices)
    create_symlinks(val_files, os.path.join(destination, 'val'))

    test_files = np.delete(files, val_indices)
    create_symlinks(test_files, os.path.join(destination, 'test'))

def create_symlinks(files, dest_dir):
    for file in files:
        os.symlink(file, os.path.join(dest_dir, os.path.basename(file)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--source', required=True,
                        help='source data directory')
    parser.add_argument('--destination', required=True,
                        help='destination data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.source, args.destination)