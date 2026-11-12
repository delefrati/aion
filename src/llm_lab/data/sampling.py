"""Batch order as a pure function of (seed, batch number), so a resumed run neither skips
nor repeats data.

A sampler that keeps state (a shuffled epoch, a RandomSampler's generator) starts over in
every new process: each session of a multi-session run replayed the same first batches.
Here batch k is computed from the seed and k alone, and a session that resumes at step s
starts at batch s * grad_accum_steps, exactly where an uninterrupted run would be.
"""
from __future__ import annotations

import torch
from torch.utils.data import Sampler


class StepBatchSampler(Sampler[list[int]]):
    """Endless batches of dataset indices; batch k depends only on (seed, k).

    replacement=True (pretraining text: billions of windows) draws each batch uniformly
    at random, with memory O(batch). replacement=False (finetuning sets) walks a fresh
    permutation every epoch, epoch e's seeded by (seed, e).

    In a run of `world` processes, rank r takes global batches k * world + r, so the ranks
    see different data and together see what one process would.
    """

    def __init__(self, n: int, batch_size: int, seed: int, start: int = 0, world: int = 1,
                 rank: int = 0, replacement: bool = True):
        if n <= 0:
            raise ValueError("StepBatchSampler: the dataset is empty")
        self.n, self.batch_size, self.seed = n, batch_size, seed
        self.start, self.world, self.rank = start, world, rank
        self.replacement = replacement
        self._epoch, self._perm = -1, None

    def _generator(self, k: int) -> torch.Generator:
        return torch.Generator().manual_seed((self.seed * 1_000_003 + k) % (2**63))

    def batch(self, g: int) -> list[int]:
        """Global batch g."""
        if self.replacement:
            return torch.randint(self.n, (self.batch_size,), generator=self._generator(g)).tolist()
        out: list[int] = []
        pos = g * self.batch_size
        while len(out) < self.batch_size:
            epoch, i = divmod(pos, self.n)
            if epoch != self._epoch:
                self._epoch, self._perm = epoch, torch.randperm(self.n, generator=self._generator(epoch)).tolist()
            take = self._perm[i:i + self.batch_size - len(out)]
            out += take
            pos += len(take)
        return out

    def __iter__(self):
        k = self.start
        while True:
            yield self.batch(k * self.world + self.rank)
            k += 1

    def __len__(self) -> int:
        return 2**31  # endless; the training loop counts steps, not batches
