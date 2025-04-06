import gzip
import os
import pickle
import urllib.request

import numpy as np

import autograd


class MNIST(autograd.TensorDataset):
    def __init__(self, train: bool, cache_dir: str = "mnist_data"):
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mnist_data")
        os.makedirs(cache_dir, exist_ok=True)
        data = self.download_or_load(train, cache_dir)
        super().__init__(*data)

    def download_or_load(self, train: bool, cache_dir: str):
        root = "https://ossci-datasets.s3.amazonaws.com/mnist"
        if train:
            urls = [
                f"{root}/train-images-idx3-ubyte.gz",
                f"{root}/train-labels-idx1-ubyte.gz",
            ]
        else:
            urls = [
                f"{root}/t10k-images-idx3-ubyte.gz",
                f"{root}/t10k-labels-idx1-ubyte.gz",
            ]

        file_paths = []
        for url in urls:
            filename = os.path.join(cache_dir, os.path.basename(url))
            file_paths.append(filename)

            if not os.path.exists(filename):
                print(f"Downloading {url}...")
                urllib.request.urlretrieve(url, filename)

        train_str = "train" if train else "test"
        cache_file = os.path.join(cache_dir, f"mnist_{train_str}.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                images, labels = pickle.load(f)
        else:
            with gzip.open(file_paths[0], "rb") as f:
                images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28, 1) / 255.0
            with gzip.open(file_paths[1], "rb") as f:
                labels = np.frombuffer(f.read(), np.uint8, offset=8)
            with open(cache_file, "wb") as f:
                pickle.dump((images, labels), f)

        return images, labels
