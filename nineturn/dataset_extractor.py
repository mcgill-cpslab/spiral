import urllib.request as ur
import zipfile
import os
import os.path as osp
from six.moves import urllib
import errno
from tqdm import tqdm
from ogb.utils.url import makedirs, maybe_log, extract_zip

GBFACTOR = float(1 << 30)

def download_url(url, folder, log=True):
    r"""Downloads the content of an URL to a specific folder.
    Args:
        url (string): The url.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """

    filename = url.rpartition('/')[2]
    filename = filename.replace("?raw=true", "")
    path = osp.join(folder, filename)

    if osp.exists(path) and osp.getsize(path) > 0:  # pragma: no cover
        if log:
            print('Using exist file', filename)
        return path

    if log:
        print('Downloading', url)

    makedirs(folder)
    data = ur.urlopen(url)

    size = int(data.info()["Content-Length"])

    chunk_size = 1024*1024
    num_iter = int(size/chunk_size) + 2

    downloaded_size = 0

    with open(path, 'wb') as f:
        pbar = tqdm(range(num_iter))
        for i in pbar:
            chunk = data.read(chunk_size)
            downloaded_size += len(chunk)
            pbar.set_description("Downloaded {:.2f} GB".format(float(downloaded_size)/GBFACTOR))
            f.write(chunk)

    return path

if __name__ == '__main__':
    folder = osp.join("C:", "Users", "baili", "Downloads")
    path = download_url("https://github.com/CooKey-Monster/Spiral-Kaggle-Dataset/blob/main/archive.zip?raw=true", folder)
    extract_zip(path, folder)