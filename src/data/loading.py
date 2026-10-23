from collections import OrderedDict
from dataclasses import fields
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import BrainsetsDataset
from src.training.sampling import collate_samples, pair_indices, sample_pair


def build_dataset(config, cohort=None):
    return BrainsetsDataset(config, cohort)


class PairDataset(Dataset):
    def __init__(self, dataset, sampling, training=True, num_workers=0, max_open_sessions=1):
        self.dataset, self.sampling, self.training = dataset, sampling, training
        if max_open_sessions < 1 or int(max_open_sessions) != max_open_sessions:
            raise ValueError("max_open_sessions must be a positive integer")
        self.max_open_sessions = int(max_open_sessions)
        self.sessions = OrderedDict()

    def close(self):
        while self.sessions:
            self.sessions.popitem()[1].close()

    def session(self, key):
        if key not in self.sessions:
            if len(self.sessions) >= self.max_open_sessions:
                self.sessions.popitem(last=False)[1].close()
            self.sessions[key] = self.dataset.load_session(*key)
        self.sessions.move_to_end(key)
        return self.sessions[key]

    def indices(self, keys, seed):
        rng = np.random.default_rng(seed)
        indices = []
        for key in keys:
            session = self.session(key)
            if session.n_units < 2 or session.t_end - session.t_start < self.sampling.duration:
                warnings.warn(f"Skipping {key}: need two units and a full window", stacklevel=2)
                continue
            indices.extend(pair_indices(
                {key: session}, [key], rng, self.sampling, random_offset=self.training,
                min_distance=float(self.sampling.min_distance),
            ))
        self.close()
        if self.training:
            rng.shuffle(indices)
        return indices

    def __getitem__(self, index):
        return sample_pair(self.session(index.key), index, self.sampling, self.training)


def collate_pairs(samples):
    batch = collate_samples(samples)
    return {field.name: torch.from_numpy(getattr(batch, field.name)) for field in fields(batch)}


def make_loader(dataset, indices, batch_size, num_workers=0, pin_memory=False, seed=0,
                prefetch_factor=1):
    dataset.close()
    # Spawn avoids forking a process that has already initialized CUDA.
    worker_options = ({"multiprocessing_context": "spawn", "prefetch_factor": prefetch_factor}
                      if num_workers else {})
    return DataLoader(
        dataset, sampler=indices, batch_size=batch_size, collate_fn=collate_pairs,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
        generator=torch.Generator().manual_seed(seed),
        **worker_options,
    )
