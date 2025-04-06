import itertools

import numpy as np


class Dataset:
    def __iter__(self):
        raise NotImplementedError("Must implement __iter__ method.")


class TensorDataset(Dataset):
    def __init__(self, *tensors):
        self.tensors = tensors

    def __len__(self):
        return len(self.tensors[0])

    def __getitem__(self, idx):
        return tuple(tensor[idx] for tensor in self.tensors)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]


class DataLoader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shufle_buffer_size: int = 0,
        repeat: bool = False,
        drop_last: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shufle_buffer_size = shufle_buffer_size
        self.repeat = repeat
        self.drop_last = drop_last

    def __iter__(self):
        it = self._iterator(self.dataset)
        if self.shufle_buffer_size > 0:
            it = self._shuffle_buffer(it)
        it = self._batch(it)
        yield from it

    def _iterator(self, dataset):
        if self.repeat:
            while True:
                yield from iter(dataset)
        else:
            yield from iter(dataset)

    def _shuffle_buffer(self, iterator):
        buffer = list(itertools.islice(iterator, self.shufle_buffer_size))
        while buffer:
            index = np.random.randint(len(buffer))
            buffer[index], buffer[-1] = buffer[-1], buffer[index]
            yield buffer.pop()
            try:
                buffer.append(next(iterator))
            except StopIteration:
                continue

    def _batch(self, iterator):
        batch = []
        for item in iterator:
            batch.append(item)
            if len(batch) == self.batch_size:
                collated = self._collate(batch)
                yield collated
                batch = []
        if not self.drop_last and batch:
            collated = self._collate(batch)
            yield collated

    def _collate(self, batch):
        transposed = zip(*batch)
        return tuple(np.array(samples) for samples in transposed)
