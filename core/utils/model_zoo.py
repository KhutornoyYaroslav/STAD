import os
import sys
import torch
from urllib import parse, request
from core.utils.dist_util import is_main_process, synchronize
try:
    from torch.hub import download_url_to_file
    from torch.hub import urlparse
    from torch.hub import HASH_REGEX
except ImportError:
    from torch.utils.model_zoo import download_url_to_file
    from torch.utils.model_zoo import urlparse
    from torch.utils.model_zoo import HASH_REGEX


def cache_url(url, model_dir=None, progress=True):
    if model_dir is None:
        torch_home = os.path.expanduser(os.getenv("TORCH_HOME", "~/.torch"))
        model_dir = os.getenv("TORCH_MODEL_ZOO", os.path.join(torch_home, "models"))

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if filename == "model_final.pkl":
        filename = parts.path.replace("/", "_")
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file) and is_main_process():
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = HASH_REGEX.search(filename)
        if hash_prefix is not None:
            hash_prefix = hash_prefix.group(1)
            if len(hash_prefix) < 6:
                hash_prefix = None
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)
    synchronize()
    return cached_file


def load_state_dict_from_url(url, map_location='cpu'):
    cached_file = cache_url(url)
    return torch.load(cached_file, map_location=map_location)


def is_url(url: str, try_download: bool = False):
    try:
        url = str(url)
        result = parse.urlparse(url)
        assert all([result.scheme, result.netloc])
        if try_download:
            with request.urlopen(url) as response:
                return response.getcode() == 200
        return True
    except Exception:
        return False


def load_state_dict(url_or_file: str, map_location='cpu'):
    if is_url(url_or_file):
        return torch.load(cache_url(url_or_file), map_location=map_location)
    elif os.path.isfile(url_or_file):
        return torch.load(url_or_file, map_location=map_location)
    else:
        raise ValueError(f"'url_or_file' must be url or path to file")
