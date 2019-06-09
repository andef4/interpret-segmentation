import requests
import os
import inspect
from pathlib import Path
import shutil
import gzip
import tarfile
import sys


if __name__ == '__main__':
    path = Path(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))

    out_file = path / 'testnet.pth'
    if out_file.exists():
        print('testnet.pth already exists')
        sys.exit(1)
    response = requests.get(
        'https://github.com/andef4/interpret-segmentation/releases/download/v1/testnet.pth.gz', stream=True)
    temp_file = path / 'testnet.pth.gz'
    if temp_file.exists():
        temp_file.unlink()
    with open(temp_file, 'wb') as f:
        shutil.copyfileobj(response.raw, f)
    with gzip.open(temp_file, 'r') as fin, open(path / 'testnet.pth', 'wb') as fout:
        shutil.copyfileobj(fin, fout)
    temp_file.unlink()

    response = requests.get(
        'https://github.com/andef4/interpret-segmentation/releases/download/v1/testnet.tar.gz', stream=True)
    temp_file = path / 'testnet.tar.gz'
    if temp_file.exists():
        temp_file.unlink()
    with open(temp_file, 'wb') as f:
        shutil.copyfileobj(response.raw, f)
    tar = tarfile.open(temp_file)
    tar.close()
    os.rename(path / 'testnet', path / 'dataset')
    temp_file.unlink()
