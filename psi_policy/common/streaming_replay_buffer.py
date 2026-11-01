from typing import Union, Dict, Optional
import zarr
import numpy as np
from functools import cached_property
import numbers
from psi_policy.common.replay_buffer import ReplayBuffer

class ZarrImageReference:
    """Lazy reference to an image array."""
    def __init__(self, zarr_array):
        self.zarr_array = zarr_array
        self.shape = zarr_array.shape
        self.dtype = zarr_array.dtype
        self.chunks = zarr_array.chunks

    def __getitem__(self, idx):
        return self.zarr_array[idx]

    def __len__(self):
        return len(self.zarr_array)

class StreamingReplayBuffer(ReplayBuffer):
    """Replay buffer with lazy image loading for large datasets."""
    def __init__(self, root=None):
        self._data = None
        self._meta = None
        self._episode_ends = None
        self.zarr_path = None

    @classmethod
    def copy_from_path(cls, path, keys=None):
        """Open a streaming replay buffer from a Zarr store."""
        buffer = cls()
        buffer.zarr_path = path


        src = zarr.open(path, mode='r')


        buffer._meta = dict()
        if 'meta' in src:
            for key, value in src['meta'].items():
                if len(value.shape) == 0:
                    buffer._meta[key] = np.array(value)
                else:
                    buffer._meta[key] = value[:]


        buffer._data = dict()
        if 'data' not in src:
            src_data = src
        else:
            src_data = src['data']

        if keys is None:
            keys = list(src_data.keys())

        for key in keys:
            arr = src_data[key]
            # Keep image arrays on disk and load low-dimensional arrays into memory.
            # Identify camera arrays by their key or by a four-dimensional uint8 layout.

            is_image = (
                key.startswith('rgb_')
                or key.startswith('cam_')
                or (getattr(arr, 'ndim', 0) >= 4 and arr.dtype == np.uint8)
            )
            if is_image:
                buffer._data[key] = ZarrImageReference(arr)
            else:
                buffer._data[key] = arr[:]

        return buffer

    @property
    def data(self):
        """Return the data mapping."""
        return self._data

    @property
    def meta(self):
        """Return the metadata mapping."""
        return self._meta

    @property
    def episode_ends(self):
        """Return cumulative episode end indices."""
        if self._episode_ends is None:
            if 'episode_ends' in self.meta:
                self._episode_ends = self.meta['episode_ends']
            else:

                first_data = next(iter(self.data.values()))
                self._episode_ends = np.array([len(first_data)])
        return self._episode_ends

    def get_episode(self, idx, copy=False):
        """Read one episode."""
        idx = list(range(len(self.episode_ends)))[idx]
        start_idx = 0
        if idx > 0:
            start_idx = self.episode_ends[idx-1]
        end_idx = self.episode_ends[idx]

        result = dict()
        for key, value in self.data.items():
            x = value[start_idx:end_idx]
            if copy and isinstance(value, np.ndarray):
                x = x.copy()
            result[key] = x
        return result

    def get_steps_slice(self, start, stop, step=None, copy=False):
        """Read a range of steps."""
        _slice = slice(start, stop, step)
        result = dict()
        for key, value in self.data.items():
            x = value[_slice]
            if copy and isinstance(value, np.ndarray):
                x = x.copy()
            result[key] = x
        return result

    def __len__(self):
        """Return the total number of steps."""
        if len(self.episode_ends) == 0:
            return 0
        return int(self.episode_ends[-1])

    def __getitem__(self, key):
        """Return the array for a key."""
        return self.data[key]

    def keys(self):
        """Return all data keys."""
        return self.data.keys()

    def values(self):
        """Return all data arrays."""
        return self.data.values()

    def items(self):
        """Return all data items."""
        return self.data.items()

    def __contains__(self, key):
        """Check whether a data key exists."""
        return key in self.data

    @property
    def n_episodes(self):
        """Return the number of episodes."""
        return len(self.episode_ends)

    def get_episode_lengths(self):
        """Return the length of each episode."""
        ends = self.episode_ends[:]
        starts = np.concatenate([[0], ends[:-1]])
        lengths = ends - starts
        return lengths
