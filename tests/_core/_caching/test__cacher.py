from epochalyst._core._caching._cacher import _Cacher
import numpy as np
import dask.dataframe as dd
import pandas as pd
import dask.array as da
from tests.util import remove_cache_files
import pytest


class Test_Cacher:
    def test_cacher_init(self):
        c = _Cacher()
        assert c is not None

    # _cache_exists
    def test__cache_exists_no_cache_args(self):
        c = _Cacher()
        assert c._cache_exists("test") is False

    def test__cache_exists_no_storage_type(self):
        c = _Cacher()
        with pytest.raises(ValueError):
            c._cache_exists("test", {"storage_path": "test"})

    def test__cache_exists_storage_type_npy(self):
        c = _Cacher()
        assert (
            c._cache_exists(
                "test", {"storage_type": ".npy", "storage_path": "tests/cache"}
            )
            is False
        )

    def test__cache_exists_storage_type_npy_exists(self):
        c = _Cacher()
        with open("tests/cache/test.npy", "w") as f:
            f.write("test")
        assert (
            c._cache_exists(
                "test", {"storage_type": ".npy", "storage_path": "tests/cache"}
            )
            is True
        )
        remove_cache_files()

    def test__cache_exists_storage_type_parquet(self):
        c = _Cacher()
        assert (
            c._cache_exists(
                "test", {"storage_type": ".parquet", "storage_path": "tests/cache"}
            )
            is False
        )

    def test__cache_exists_storage_type_parquet_exists(self):
        c = _Cacher()
        with open("tests/cache/test.parquet", "w") as f:
            f.write("test")
        assert (
            c._cache_exists(
                "test", {"storage_type": ".parquet", "storage_path": "tests/cache"}
            )
            is True
        )
        remove_cache_files()

    def test__cache_exists_storage_type_csv(self):
        c = _Cacher()
        assert (
            c._cache_exists(
                "test", {"storage_type": ".csv", "storage_path": "tests/cache"}
            )
            is False
        )

    def test__cache_exists_storage_type_csv_exists(self):
        c = _Cacher()
        with open("tests/cache/test.csv", "w") as f:
            f.write("test")
        assert (
            c._cache_exists(
                "test", {"storage_type": ".csv", "storage_path": "tests/cache"}
            )
            is True
        )
        remove_cache_files()

    def test__cache_exists_storage_type_npy_stack(self):
        c = _Cacher()
        assert (
            c._cache_exists(
                "test", {"storage_type": ".npy_stack", "storage_path": "tests/cache"}
            )
            is False
        )

    def test__cache_exists_storage_type_npy_stack_exists(self):
        c = _Cacher()
        with open("tests/cache/test", "w") as f:
            f.write("test")
        assert (
            c._cache_exists(
                "test", {"storage_type": ".npy_stack", "storage_path": "tests/cache"}
            )
            is True
        )
        remove_cache_files()

    def test__cache_exists_storage_type_unsupported(self):
        c = _Cacher()
        assert (
            c._cache_exists(
                "test", {"storage_type": ".new_type", "storage_path": "tests/cache"}
            )
            is False
        )

    # _store_cache
    def test__store_cache_no_cache_args(self):
        c = _Cacher()
        with pytest.raises(ValueError):
            c._store_cache("test", {})

    def test__store_cache_no_storage_type(self):
        c = _Cacher()
        with pytest.raises(ValueError):
            c._store_cache("test", "test", {"storage_path": "test"})

    def test__store_cache_no_storage_path(self):
        c = _Cacher()
        with pytest.raises(ValueError):
            c._store_cache("test", "test", {"storage_type": ".npy"})

    def test__store_cache_no_output_data_type(self):
        c = _Cacher()
        with pytest.raises(ValueError):
            c._store_cache(
                "test", "test", {"storage_type": ".npy", "storage_path": "tests/cache"}
            )

    # storage type .npy
    def test__store_cache_storage_type_npy_output_data_type_numpy_array(self):
        c = _Cacher()
        c._store_cache(
            "test",
            "test",
            {
                "storage_type": ".npy",
                "storage_path": "tests/cache",
                "output_data_type": "numpy_array",
            },
        )
        assert (
            c._cache_exists(
                "test", {"storage_type": ".npy", "storage_path": "tests/cache"}
            )
            is True
        )
        remove_cache_files()

    def test__store_cache_storage_type_npy_output_data_type_dask_array(self):
        c = _Cacher()
        # Create dask array
        x = da.ones((1000, 1000), chunks=(100, 100))
        c._store_cache(
            "test",
            x,
            {
                "storage_type": ".npy",
                "storage_path": "tests/cache",
                "output_data_type": "dask_array",
            },
        )
        assert (
            c._cache_exists(
                "test", {"storage_type": ".npy", "storage_path": "tests/cache"}
            )
            is True
        )
        remove_cache_files()

    def test__store_cache_storage_type_npy_output_data_type_unsupported(self):
        c = _Cacher()
        with pytest.raises(ValueError):
            c._store_cache(
                "test",
                "test",
                {
                    "storage_type": ".npy",
                    "storage_path": "tests/cache",
                    "output_data_type": "pandas_dataframe",
                },
            )

    # storage type .parquet
    def test__store_cache_storage_type_parquet_output_data_type_pandas_dataframe(self):
        c = _Cacher()
        # Pandas dataframe
        data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        c._store_cache(
            "test",
            data,
            {
                "storage_type": ".parquet",
                "storage_path": "tests/cache",
                "output_data_type": "pandas_dataframe",
            },
        )
        assert (
            c._cache_exists(
                "test", {"storage_type": ".parquet", "storage_path": "tests/cache"}
            )
            is True
        )
        remove_cache_files()

    def test__store_cache_storage_type_parquet_output_data_type_dask_dataframe(self):
        c = _Cacher()
        # Dask dataframe
        data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        data = dd.from_pandas(data, npartitions=2)
        c._store_cache(
            "test",
            data,
            {
                "storage_type": ".parquet",
                "storage_path": "tests/cache",
                "output_data_type": "dask_dataframe",
            },
        )
        assert (
            c._cache_exists(
                "test", {"storage_type": ".parquet", "storage_path": "tests/cache"}
            )
            is True
        )
        remove_cache_files()

    def test__store_cache_storage_type_parquet_output_data_type_numpy_array(self):
        c = _Cacher()
        # Numpy array
        data = np.array([1, 2, 3])
        c._store_cache(
            "test",
            data,
            {
                "storage_type": ".parquet",
                "storage_path": "tests/cache",
                "output_data_type": "numpy_array",
            },
        )
        assert (
            c._cache_exists(
                "test", {"storage_type": ".parquet", "storage_path": "tests/cache"}
            )
            is True
        )
        remove_cache_files()

    def test__store_cache_storage_type_parquet_output_data_type_dask_array(self):
        c = _Cacher()
        # Dask array
        data = da.ones((1000, 1000), chunks=(100, 100))
        c._store_cache(
            "test",
            data,
            {
                "storage_type": ".parquet",
                "storage_path": "tests/cache",
                "output_data_type": "dask_array",
            },
        )
        assert (
            c._cache_exists(
                "test", {"storage_type": ".parquet", "storage_path": "tests/cache"}
            )
            is True
        )
        remove_cache_files()

    def test__store_cache_storage_type_parquet_output_data_type_unsupported(self):
        c = _Cacher()
        with pytest.raises(ValueError):
            c._store_cache(
                "test",
                "test",
                {
                    "storage_type": ".parquet",
                    "storage_path": "tests/cache",
                    "output_data_type": "new_type",
                },
            )

    # storage type .csv
    def test__store_cache_storage_type_csv_output_data_type_pandas_dataframe(self):
        c = _Cacher()
        # Pandas dataframe
        data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        c._store_cache(
            "test",
            data,
            {
                "storage_type": ".csv",
                "storage_path": "tests/cache",
                "output_data_type": "pandas_dataframe",
            },
        )
        assert (
            c._cache_exists(
                "test", {"storage_type": ".csv", "storage_path": "tests/cache"}
            )
            is True
        )
        remove_cache_files()

    def test__store_cache_storage_type_csv_output_data_type_dask_dataframe(self):
        c = _Cacher()
        # Dask dataframe
        data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        data = dd.from_pandas(data, npartitions=2)
        c._store_cache(
            "test",
            data,
            {
                "storage_type": ".csv",
                "storage_path": "tests/cache",
                "output_data_type": "dask_dataframe",
            },
        )
        assert (
            c._cache_exists(
                "test", {"storage_type": ".csv", "storage_path": "tests/cache"}
            )
            is True
        )
        remove_cache_files()

    def test__store_cache_storage_type_csv_output_data_type_unsupported(self):
        c = _Cacher()
        with pytest.raises(ValueError):
            c._store_cache(
                "test",
                "test",
                {
                    "storage_type": ".csv",
                    "storage_path": "tests/cache",
                    "output_data_type": "new_type",
                },
            )

    # storage type .npy_stack
    def test__store_cache_storage_type_npy_stack_output_data_type_dask_array(self):
        c = _Cacher()
        # Dask array
        data = da.ones((1000, 1000), chunks=(100, 100))
        c._store_cache(
            "test",
            data,
            {
                "storage_type": ".npy_stack",
                "storage_path": "tests/cache",
                "output_data_type": "dask_array",
            },
        )
        assert (
            c._cache_exists(
                "test", {"storage_type": ".npy_stack", "storage_path": "tests/cache"}
            )
            is True
        )

    def test__store_cache_storage_type_npy_stack_output_data_type_unsupported(self):
        c = _Cacher()
        with pytest.raises(ValueError):
            c._store_cache(
                "test",
                "test",
                {
                    "storage_type": ".npy_stack",
                    "storage_path": "tests/cache",
                    "output_data_type": "numpy_array",
                },
            )

    def test__store_cache_storage_type_unsupported(self):
        c = _Cacher()
        with pytest.raises(ValueError):
            c._store_cache(
                "test",
                "test",
                {
                    "storage_type": ".new_type",
                    "storage_path": "tests/cache",
                    "output_data_type": "numpy_array",
                },
            )

    # _get_cache
    def test__get_cache_no_cache_args(self):
        c = _Cacher()
        with pytest.raises(ValueError):
            c._get_cache("test", {})

    def test__get_cache_no_storage_type(self):
        c = _Cacher()
        with pytest.raises(ValueError):
            c._get_cache("test", {"storage_path": "test"})

    def test__get_cache_no_storage_path(self):
        c = _Cacher()
        with pytest.raises(ValueError):
            c._get_cache("test", {"storage_type": ".npy"})

    def test__get_cache_no_output_data_type(self):
        c = _Cacher()
        with pytest.raises(ValueError):
            c._get_cache(
                "test", {"storage_type": ".npy", "storage_path": "tests/cache"}
            )

    # storage type .npy
    def test__get_cache_storage_type_npy_output_data_type_numpy_array(self):
        c = _Cacher()
        c._store_cache(
            "test",
            "test",
            {
                "storage_type": ".npy",
                "storage_path": "tests/cache",
                "output_data_type": "numpy_array",
            },
        )
        assert (
            c._get_cache(
                "test",
                {
                    "storage_type": ".npy",
                    "storage_path": "tests/cache",
                    "output_data_type": "numpy_array",
                },
            )
            == "test"
        )
        remove_cache_files()

    def test__get_cache_storage_type_npy_output_data_type_dask_array(self):
        c = _Cacher()
        # Create dask array
        x = da.ones((1000, 1000), chunks=(100, 100))
        c._store_cache(
            "test",
            x,
            {
                "storage_type": ".npy",
                "storage_path": "tests/cache",
                "output_data_type": "dask_array",
            },
        )
        # Check all elements are equal
        assert (
            c._get_cache(
                "test",
                {
                    "storage_type": ".npy",
                    "storage_path": "tests/cache",
                    "output_data_type": "dask_array",
                },
            )
            .compute()
            .all()
            == x.compute().all()
        )
        remove_cache_files()

    def test__get_cache_storage_type_npy_output_data_type_unsupported(self):
        c = _Cacher()
        with pytest.raises(ValueError):
            c._get_cache(
                "test",
                {
                    "storage_type": ".npy",
                    "storage_path": "tests/cache",
                    "output_data_type": "pandas_dataframe",
                },
            )

    # storage type .parquet
    def test__get_cache_storage_type_parquet_output_data_type_pandas_dataframe(self):
        c = _Cacher()
        # Pandas dataframe
        data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        c._store_cache(
            "test",
            data,
            {
                "storage_type": ".parquet",
                "storage_path": "tests/cache",
                "output_data_type": "pandas_dataframe",
            },
        )
        assert c._get_cache(
            "test",
            {
                "storage_type": ".parquet",
                "storage_path": "tests/cache",
                "output_data_type": "pandas_dataframe",
            },
        ).equals(data)
        remove_cache_files()

    def test__get_cache_storage_type_parquet_output_data_type_dask_dataframe(self):
        c = _Cacher()
        # Dask dataframe
        data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        data = dd.from_pandas(data, npartitions=2)
        c._store_cache(
            "test",
            data,
            {
                "storage_type": ".parquet",
                "storage_path": "tests/cache",
                "output_data_type": "dask_dataframe",
            },
        )
        assert (
            c._get_cache(
                "test",
                {
                    "storage_type": ".parquet",
                    "storage_path": "tests/cache",
                    "output_data_type": "dask_dataframe",
                },
            )
            .compute()
            .equals(data.compute())
        )
        remove_cache_files()

    def test__get_cache_storage_type_parquet_output_data_type_numpy_array(self):
        c = _Cacher()
        # Numpy array
        data = np.array([1, 2, 3])
        c._store_cache(
            "test",
            data,
            {
                "storage_type": ".parquet",
                "storage_path": "tests/cache",
                "output_data_type": "numpy_array",
            },
        )
        get_cache = c._get_cache(
            "test",
            {
                "storage_type": ".parquet",
                "storage_path": "tests/cache",
                "output_data_type": "numpy_array",
            },
        )
        assert get_cache.all() == data.all()
        remove_cache_files()

    def test__get_cache_storage_type_parquet_output_data_type_dask_array(self):
        c = _Cacher()
        # Dask array
        data = da.ones((1000, 1000), chunks=(100, 100))
        c._store_cache(
            "test",
            data,
            {
                "storage_type": ".parquet",
                "storage_path": "tests/cache",
                "output_data_type": "dask_array",
            },
        )
        get_cache = c._get_cache(
            "test",
            {
                "storage_type": ".parquet",
                "storage_path": "tests/cache",
                "output_data_type": "dask_array",
            },
        )
        assert get_cache.compute().all() == data.compute().all()
        remove_cache_files()

    def test__get_cache_storage_type_parquet_output_data_type_unsupported(self):
        c = _Cacher()
        with pytest.raises(ValueError):
            c._get_cache(
                "test",
                {
                    "storage_type": ".parquet",
                    "storage_path": "tests/cache",
                    "output_data_type": "new_type",
                },
            )

    # storage type .csv
    def test__get_cache_storage_type_csv_output_data_type_pandas_dataframe(self):
        c = _Cacher()
        # Pandas dataframe
        data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        c._store_cache(
            "test",
            data,
            {
                "storage_type": ".csv",
                "storage_path": "tests/cache",
                "output_data_type": "pandas_dataframe",
            },
        )
        get_cache = c._get_cache(
            "test",
            {
                "storage_type": ".csv",
                "storage_path": "tests/cache",
                "output_data_type": "pandas_dataframe",
            },
        )
        assert get_cache.equals(data)
        remove_cache_files()

    def test__get_cache_storage_type_csv_output_data_type_dask_dataframe(self):
        c = _Cacher()
        # Dask dataframe
        data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        data = dd.from_pandas(data, npartitions=2)
        c._store_cache(
            "test",
            data,
            {
                "storage_type": ".csv",
                "storage_path": "tests/cache",
                "output_data_type": "dask_dataframe",
            },
        )
        get_cache = c._get_cache(
            "test",
            {
                "storage_type": ".csv",
                "storage_path": "tests/cache",
                "output_data_type": "dask_dataframe",
            },
        )
        assert get_cache.compute().reset_index(drop=True).equals(data.compute())
        remove_cache_files()

    def test__get_cache_storage_type_csv_output_data_type_unsupported(self):
        c = _Cacher()
        with pytest.raises(ValueError):
            c._get_cache(
                "test",
                {
                    "storage_type": ".csv",
                    "storage_path": "tests/cache",
                    "output_data_type": "new_type",
                },
            )

    # storage type .npy_stack
    def test__get_cache_storage_type_npy_stack_output_data_type_dask_array(self):
        c = _Cacher()
        # Dask array
        data = da.ones((1000, 1000), chunks=(100, 100))
        c._store_cache(
            "test",
            data,
            {
                "storage_type": ".npy_stack",
                "storage_path": "tests/cache",
                "output_data_type": "dask_array",
            },
        )
        get_cache = c._get_cache(
            "test",
            {
                "storage_type": ".npy_stack",
                "storage_path": "tests/cache",
                "output_data_type": "dask_array",
            },
        )
        assert get_cache.compute().all() == data.compute().all()
        remove_cache_files()

    def test__get_cache_storage_type_npy_stack_output_data_type_unsupported(self):
        c = _Cacher()
        with pytest.raises(ValueError):
            c._get_cache(
                "test",
                {
                    "storage_type": ".npy_stack",
                    "storage_path": "tests/cache",
                    "output_data_type": "numpy_array",
                },
            )

    def test__get_cache_storage_type_unsupported(self):
        c = _Cacher()
        with pytest.raises(ValueError):
            c._get_cache(
                "test",
                {
                    "storage_type": ".new_type",
                    "storage_path": "tests/cache",
                    "output_data_type": "numpy_array",
                },
            )
