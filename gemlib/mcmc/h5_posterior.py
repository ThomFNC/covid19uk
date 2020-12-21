"""Class for writing posterior samples"""

import h5py


def _maybe_tf_dtype(dtype):
    if hasattr(dtype, "as_numpy_dtype"):
        return dtype.as_numpy_dtype
    return dtype


def _maybe_to_numpy(val):
    if hasattr(val, "numpy"):
        return val.numpy()
    return val


class Posterior:
    def __init__(self, filename, sample_dict, results_dict, num_samples):

        self._num_samples = num_samples
        self._file = h5py.File(
            filename,
            "w",
            rdcc_nbytes=1024 ** 2 * 400,
            rdcc_nslots=100000,
            libver="latest",
        )
        self._file.swmr_mode = True

        self._sample_group = self._file.create_group("samples")
        self._create_data_tree(sample_dict, self._sample_group)

        self._results_group = self._file.create_group("results")
        self._create_data_tree(results_dict, self._results_group)

    def __del__(self):
        self._file.close()

    def __getitem__(self, path):
        return self._file[path]

    def _create_data_tree(self, data_dict, h5dataset):
        for k, v in data_dict.items():
            if isinstance(v, dict):
                h5group = h5dataset.create_group(k)
                self._create_data_tree(v, h5group)
            else:
                if isinstance(v, tuple):
                    data = v[0]
                    kwargs = dict(
                        chunks=v[1],
                        compression="szip",
                        compression_opts=("nn", 16),
                    )
                else:
                    data = v
                    kwargs = {}

                h5dataset.create_dataset(
                    k,
                    (self._num_samples,) + data.shape[1:],
                    dtype=_maybe_tf_dtype(data.dtype),
                    **kwargs,
                )

    def _write(self, sample_dict, dset, first_dim_offset=0):
        for k, v in sample_dict.items():
            if isinstance(v, dict):
                self._write(v, dset[k], first_dim_offset)
            else:
                s = slice(first_dim_offset, first_dim_offset + v.shape[0])
                dset[k][s, ...] = _maybe_to_numpy(v)

    def write_samples(self, samples_dict, first_dim_offset=0):
        self._write(samples_dict, self._sample_group, first_dim_offset)
        self._file.flush()

    def write_results(self, results_dict, first_dim_offset=0):
        self._write(results_dict, self._results_group, first_dim_offset)
        self._file.flush()
