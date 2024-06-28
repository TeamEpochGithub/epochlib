import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import polars as pl
import pytest

from epochalyst.caching.cacher import Cacher
from epochalyst.logging.logger import Logger
from tests.constants import TEMP_DIR


class Implemented_Cacher(Cacher, Logger):
    pass


class Test_Cacher:
    cache_path = TEMP_DIR

    @pytest.fixture(autouse=True)
    def run_always(self, setup_temp_dir):
        pass

    def test_cacher_init(self):
        c = Implemented_Cacher()
        assert c is not None

    # _cache_exists
    def test__cache_exists_no_cache_args(self):
        c = Implemented_Cacher()
        assert c.cache_exists("test") is False

    def test__cache_exists_no_storage_type(self):
        c = Implemented_Cacher()
        with pytest.raises(ValueError):
            c.cache_exists("test", {"storage_path": "test"})

    def test__cache_exists_storage_type_npy(self):
        c = Implemented_Cacher()
        assert (
            c.cache_exists(
                "test", {"storage_type": ".npy", "storage_path": f"{self.cache_path}"}
            )
            is False
        )

    def test__cache_exists_storage_type_npy_exists(self):
        c = Implemented_Cacher()
        with open(self.cache_path / "test.npy", "w") as f:
            f.write("test")
        assert (
            c.cache_exists(
                "test", {"storage_type": ".npy", "storage_path": f"{self.cache_path}"}
            )
            is True
        )

    def test__cache_exists_storage_type_parquet(self):
        c = Implemented_Cacher()
        assert (
            c.cache_exists(
                "test",
                {"storage_type": ".parquet", "storage_path": f"{self.cache_path}"},
            )
            is False
        )

    def test__cache_exists_storage_type_parquet_exists(self):
        c = Implemented_Cacher()
        with open(self.cache_path / "test.parquet", "w") as f:
            f.write("test")
        assert (
            c.cache_exists(
                "test",
                {"storage_type": ".parquet", "storage_path": f"{self.cache_path}"},
            )
            is True
        )

    def test__cache_exists_storage_type_csv(self):
        c = Implemented_Cacher()
        assert (
            c.cache_exists(
                "test", {"storage_type": ".csv", "storage_path": f"{self.cache_path}"}
            )
            is False
        )

    def test__cache_exists_storage_type_csv_exists(self):
        c = Implemented_Cacher()
        with open(self.cache_path / "test.csv", "w") as f:
            f.write("test")
        assert (
            c.cache_exists(
                "test", {"storage_type": ".csv", "storage_path": f"{self.cache_path}"}
            )
            is True
        )

    def test__cache_exists_storage_type_npy_stack(self):
        c = Implemented_Cacher()
        assert (
            c.cache_exists(
                "test",
                {"storage_type": ".npy_stack", "storage_path": f"{self.cache_path}"},
            )
            is False
        )

    def test__cache_exists_storage_type_npy_stack_exists(self):
        c = Implemented_Cacher()
        with open(self.cache_path / "test", "w") as f:
            f.write("test")
        assert (
            c.cache_exists(
                "test",
                {"storage_type": ".npy_stack", "storage_path": f"{self.cache_path}"},
            )
            is True
        )

    def test__cache_exists_storage_type_pkl(self):
        c = Implemented_Cacher()
        assert (
            c.cache_exists(
                "test", {"storage_type": ".pkl", "storage_path": f"{self.cache_path}"}
            )
            is False
        )

    def test__cache_exists_storage_type_pkl_exists(self):
        c = Implemented_Cacher()
        with open(self.cache_path / "test.pkl", "w") as f:
            f.write("test")
        assert (
            c.cache_exists(
                "test", {"storage_type": ".pkl", "storage_path": f"{self.cache_path}"}
            )
            is True
        )

    def test__cache_exists_storage_type_unsupported(self):
        c = Implemented_Cacher()
        assert (
            c.cache_exists(
                "test",
                {"storage_type": ".new_type", "storage_path": f"{self.cache_path}"},
            )
            is False
        )

    # _store_cache
    def test__store_cache_no_cache_args(self):
        c = Implemented_Cacher()
        with pytest.raises(ValueError):
            c._store_cache("test", {})

    def test__store_cache_no_storage_type(self):
        c = Implemented_Cacher()
        with pytest.raises(ValueError):
            c._store_cache("test", "test", {"storage_path": "test"})

    def test__store_cache_no_storage_path(self):
        c = Implemented_Cacher()
        with pytest.raises(ValueError):
            c._store_cache("test", "test", {"storage_type": ".npy"})

    def test__store_cache_no_output_data_type(self):
        c = Implemented_Cacher()
        with pytest.raises(ValueError):
            c._store_cache(
                "test",
                "test",
                {"storage_type": ".npy", "storage_path": f"{self.cache_path}"},
            )

    # storage type .npy
    def test__store_cache_storage_type_npy_output_data_type_numpy_array(self):
        c = Implemented_Cacher()
        c._store_cache(
            "test",
            "test",
            {
                "storage_type": ".npy",
                "storage_path": f"{self.cache_path}",
                "output_data_type": "numpy_array",
            },
        )
        assert (
            c.cache_exists(
                "test", {"storage_type": ".npy", "storage_path": f"{self.cache_path}"}
            )
            is True
        )

    def test__store_cache_storage_type_npy_output_data_type_dask_array(self):
        c = Implemented_Cacher()
        # Create dask array
        x = da.ones((1000, 1000), chunks=(100, 100))
        c._store_cache(
            "test",
            x,
            {
                "storage_type": ".npy",
                "storage_path": f"{self.cache_path}",
                "output_data_type": "dask_array",
            },
        )
        assert (
            c.cache_exists(
                "test", {"storage_type": ".npy", "storage_path": f"{self.cache_path}"}
            )
            is True
        )

    def test__store_cache_storage_type_npy_output_data_type_unsupported(self):
        c = Implemented_Cacher()
        with pytest.raises(ValueError):
            c._store_cache(
                "test",
                "test",
                {
                    "storage_type": ".npy",
                    "storage_path": f"{self.cache_path}",
                    "output_data_type": "pandas_dataframe",
                },
            )

    # storage type .parquet
    def test__store_cache_storage_type_parquet_output_data_type_pandas_dataframe(self):
        c = Implemented_Cacher()
        # Pandas dataframe
        data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        c._store_cache(
            "test",
            data,
            {
                "storage_type": ".parquet",
                "storage_path": f"{self.cache_path}",
                "output_data_type": "pandas_dataframe",
            },
        )
        assert (
            c.cache_exists(
                "test",
                {"storage_type": ".parquet", "storage_path": f"{self.cache_path}"},
            )
            is True
        )

    def test__store_cache_storage_type_parquet_output_data_type_dask_dataframe(self):
        c = Implemented_Cacher()
        # Dask dataframe
        data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        data = dd.from_pandas(data, npartitions=2)
        c._store_cache(
            "test",
            data,
            {
                "storage_type": ".parquet",
                "storage_path": f"{self.cache_path}",
                "output_data_type": "dask_dataframe",
            },
        )
        assert (
            c.cache_exists(
                "test",
                {"storage_type": ".parquet", "storage_path": f"{self.cache_path}"},
            )
            is True
        )

    def test__store_cache_storage_type_parquet_output_data_type_numpy_array(self):
        c = Implemented_Cacher()
        # Numpy array
        data = np.array([1, 2, 3])
        c._store_cache(
            "test",
            data,
            {
                "storage_type": ".parquet",
                "storage_path": f"{self.cache_path}",
                "output_data_type": "numpy_array",
            },
        )
        assert (
            c.cache_exists(
                "test",
                {"storage_type": ".parquet", "storage_path": f"{self.cache_path}"},
            )
            is True
        )

    def test__store_cache_storage_type_parquet_output_data_type_dask_array(self):
        c = Implemented_Cacher()
        # Dask array
        data = da.ones((1000, 1000), chunks=(100, 100))
        c._store_cache(
            "test",
            data,
            {
                "storage_type": ".parquet",
                "storage_path": f"{self.cache_path}",
                "output_data_type": "dask_array",
            },
        )
        assert (
            c.cache_exists(
                "test",
                {"storage_type": ".parquet", "storage_path": f"{self.cache_path}"},
            )
            is True
        )

    def test__store_cache_storage_type_parquet_output_data_type_polars_dataframe(self):
        c = Implemented_Cacher()
        # Polars dataframe
        data = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        c._store_cache(
            "test",
            data,
            {
                "storage_type": ".parquet",
                "storage_path": f"{self.cache_path}",
                "output_data_type": "polars_dataframe",
            },
        )
        assert (
            c.cache_exists(
                "test",
                {"storage_type": ".parquet", "storage_path": f"{self.cache_path}"},
            )
            is True
        )

    def test__store_cache_storage_type_parquet_output_data_type_unsupported(self):
        c = Implemented_Cacher()
        with pytest.raises(ValueError):
            c._store_cache(
                "test",
                "test",
                {
                    "storage_type": ".parquet",
                    "storage_path": f"{self.cache_path}",
                    "output_data_type": "new_type",
                },
            )

    # storage type .csv
    def test__store_cache_storage_type_csv_output_data_type_pandas_dataframe(self):
        c = Implemented_Cacher()
        # Pandas dataframe
        data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        c._store_cache(
            "test",
            data,
            {
                "storage_type": ".csv",
                "storage_path": f"{self.cache_path}",
                "output_data_type": "pandas_dataframe",
            },
        )
        assert (
            c.cache_exists(
                "test", {"storage_type": ".csv", "storage_path": f"{self.cache_path}"}
            )
            is True
        )

    def test__store_cache_storage_type_csv_output_data_type_dask_dataframe(self):
        c = Implemented_Cacher()
        # Dask dataframe
        data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        data = dd.from_pandas(data, npartitions=2)
        c._store_cache(
            "test",
            data,
            {
                "storage_type": ".csv",
                "storage_path": f"{self.cache_path}",
                "output_data_type": "dask_dataframe",
            },
        )
        assert (
            c.cache_exists(
                "test", {"storage_type": ".csv", "storage_path": f"{self.cache_path}"}
            )
            is True
        )

    def test__store_cache_storage_type_csv_output_data_type_polars_dataframe(self):
        c = Implemented_Cacher()
        # Polars dataframe
        data = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        c._store_cache(
            "test",
            data,
            {
                "storage_type": ".csv",
                "storage_path": f"{self.cache_path}",
                "output_data_type": "polars_dataframe",
            },
        )
        assert (
            c.cache_exists(
                "test", {"storage_type": ".csv", "storage_path": f"{self.cache_path}"}
            )
            is True
        )

    def test__store_cache_storage_type_csv_output_data_type_unsupported(self):
        c = Implemented_Cacher()
        with pytest.raises(ValueError):
            c._store_cache(
                "test",
                "test",
                {
                    "storage_type": ".csv",
                    "storage_path": f"{self.cache_path}",
                    "output_data_type": "new_type",
                },
            )

    # storage type .npy_stack
    def test__store_cache_storage_type_npy_stack_output_data_type_dask_array(self):
        c = Implemented_Cacher()
        # Dask array
        data = da.ones((1000, 1000), chunks=(100, 100))
        c._store_cache(
            "test",
            data,
            {
                "storage_type": ".npy_stack",
                "storage_path": f"{self.cache_path}",
                "output_data_type": "dask_array",
            },
        )
        assert (
            c.cache_exists(
                "test",
                {"storage_type": ".npy_stack", "storage_path": f"{self.cache_path}"},
            )
            is True
        )

    def test__store_cache_storage_type_npy_stack_output_data_type_unsupported(self):
        c = Implemented_Cacher()
        with pytest.raises(ValueError):
            c._store_cache(
                "test",
                "test",
                {
                    "storage_type": ".npy_stack",
                    "storage_path": f"{self.cache_path}",
                    "output_data_type": "numpy_array",
                },
            )

    def test__store_cache_storage_type_unsupported(self):
        c = Implemented_Cacher()
        with pytest.raises(ValueError):
            c._store_cache(
                "test",
                "test",
                {
                    "storage_type": ".new_type",
                    "storage_path": f"{self.cache_path}",
                    "output_data_type": "numpy_array",
                },
            )

    # storage type .pkl
    def test__store_cache_storage_type_pkl_output_data_type_pandas_dataframe(self):
        c = Implemented_Cacher()
        # Pandas dataframe
        data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        c._store_cache(
            "test",
            data,
            {
                "storage_type": ".pkl",
                "storage_path": f"{self.cache_path}",
                "output_data_type": "pandas_dataframe",
            },
        )
        assert (
            c.cache_exists(
                "test", {"storage_type": ".pkl", "storage_path": f"{self.cache_path}"}
            )
            is True
        )

    def test__store_cache_storage_type_pkl_output_data_type_dask_dataframe(self):
        c = Implemented_Cacher()
        # Dask dataframe
        data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        data = dd.from_pandas(data, npartitions=2)
        c._store_cache(
            "test",
            data,
            {
                "storage_type": ".pkl",
                "storage_path": f"{self.cache_path}",
                "output_data_type": "dask_dataframe",
            },
        )
        assert (
            c.cache_exists(
                "test", {"storage_type": ".pkl", "storage_path": f"{self.cache_path}"}
            )
            is True
        )

    # _get_cache
    def test__get_cache_no_cache_args(self):
        c = Implemented_Cacher()
        with pytest.raises(ValueError):
            c._get_cache("test", {})

    def test__get_cache_no_storage_type(self):
        c = Implemented_Cacher()
        with pytest.raises(ValueError):
            c._get_cache("test", {"storage_path": "test"})

    def test__get_cache_no_storage_path(self):
        c = Implemented_Cacher()
        with pytest.raises(ValueError):
            c._get_cache("test", {"storage_type": ".npy"})

    def test__get_cache_no_output_data_type(self):
        c = Implemented_Cacher()
        with pytest.raises(ValueError):
            c._get_cache(
                "test", {"storage_type": ".npy", "storage_path": f"{self.cache_path}"}
            )

    # storage type .npy
    def test__get_cache_storage_type_npy_output_data_type_numpy_array(self):
        c = Implemented_Cacher()
        c._store_cache(
            "test",
            "test",
            {
                "storage_type": ".npy",
                "storage_path": f"{self.cache_path}",
                "output_data_type": "numpy_array",
            },
        )
        assert (
            c._get_cache(
                "test",
                {
                    "storage_type": ".npy",
                    "storage_path": f"{self.cache_path}",
                    "output_data_type": "numpy_array",
                },
            )
            == "test"
        )

    def test__get_cache_storage_type_npy_output_data_type_dask_array(self):
        c = Implemented_Cacher()
        # Create dask array
        x = da.ones((1000, 1000), chunks=(100, 100))
        c._store_cache(
            "test",
            x,
            {
                "storage_type": ".npy",
                "storage_path": f"{self.cache_path}",
                "output_data_type": "dask_array",
            },
        )
        # Check all elements are equal
        assert (
            c._get_cache(
                "test",
                {
                    "storage_type": ".npy",
                    "storage_path": f"{self.cache_path}",
                    "output_data_type": "dask_array",
                },
            )
            .compute()
            .all()
            == x.compute().all()
        )

    def test__get_cache_storage_type_npy_output_data_type_unsupported(self):
        c = Implemented_Cacher()
        with pytest.raises(ValueError):
            c._get_cache(
                "test",
                {
                    "storage_type": ".npy",
                    "storage_path": f"{self.cache_path}",
                    "output_data_type": "pandas_dataframe",
                },
            )

    # storage type .parquet
    def test__get_cache_storage_type_parquet_output_data_type_pandas_dataframe(self):
        c = Implemented_Cacher()
        # Pandas dataframe
        data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        c._store_cache(
            "test",
            data,
            {
                "storage_type": ".parquet",
                "storage_path": f"{self.cache_path}",
                "output_data_type": "pandas_dataframe",
            },
        )
        assert c._get_cache(
            "test",
            {
                "storage_type": ".parquet",
                "storage_path": f"{self.cache_path}",
                "output_data_type": "pandas_dataframe",
            },
        ).equals(data)

    def test__get_cache_storage_type_parquet_output_data_type_dask_dataframe(self):
        c = Implemented_Cacher()
        # Dask dataframe
        data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        data = dd.from_pandas(data, npartitions=2)
        c._store_cache(
            "test",
            data,
            {
                "storage_type": ".parquet",
                "storage_path": f"{self.cache_path}",
                "output_data_type": "dask_dataframe",
            },
        )
        assert (
            c._get_cache(
                "test",
                {
                    "storage_type": ".parquet",
                    "storage_path": f"{self.cache_path}",
                    "output_data_type": "dask_dataframe",
                },
            )
            .compute()
            .equals(data.compute())
        )

    def test__get_cache_storage_type_parquet_output_data_type_numpy_array(self):
        c = Implemented_Cacher()
        # Numpy array
        data = np.array([1, 2, 3])
        c._store_cache(
            "test",
            data,
            {
                "storage_type": ".parquet",
                "storage_path": f"{self.cache_path}",
                "output_data_type": "numpy_array",
            },
        )
        get_cache = c._get_cache(
            "test",
            {
                "storage_type": ".parquet",
                "storage_path": f"{self.cache_path}",
                "output_data_type": "numpy_array",
            },
        )
        assert get_cache.all() == data.all()

    def test__get_cache_storage_type_parquet_output_data_type_dask_array(self):
        c = Implemented_Cacher()
        # Dask array
        data = da.ones((1000, 1000), chunks=(100, 100))
        c._store_cache(
            "test",
            data,
            {
                "storage_type": ".parquet",
                "storage_path": f"{self.cache_path}",
                "output_data_type": "dask_array",
            },
        )
        get_cache = c._get_cache(
            "test",
            {
                "storage_type": ".parquet",
                "storage_path": f"{self.cache_path}",
                "output_data_type": "dask_array",
            },
        )
        assert get_cache.compute().all() == data.compute().all()

    def test__get_cache_storage_type_parquet_output_data_type_polars_dataframe(self):
        c = Implemented_Cacher()
        # Polars dataframe
        data = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        c._store_cache(
            "test",
            data,
            {
                "storage_type": ".parquet",
                "storage_path": f"{self.cache_path}",
                "output_data_type": "polars_dataframe",
            },
        )
        get_cache = c._get_cache(
            "test",
            {
                "storage_type": ".parquet",
                "storage_path": f"{self.cache_path}",
                "output_data_type": "polars_dataframe",
            },
        )
        assert data.equals(get_cache)

    def test__get_cache_storage_type_parquet_output_data_type_unsupported(self):
        c = Implemented_Cacher()
        with pytest.raises(ValueError):
            c._get_cache(
                "test",
                {
                    "storage_type": ".parquet",
                    "storage_path": f"{self.cache_path}",
                    "output_data_type": "new_type",
                },
            )

    # storage type .csv
    def test__get_cache_storage_type_csv_output_data_type_pandas_dataframe(self):
        c = Implemented_Cacher()
        # Pandas dataframe
        data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        c._store_cache(
            "test",
            data,
            {
                "storage_type": ".csv",
                "storage_path": f"{self.cache_path}",
                "output_data_type": "pandas_dataframe",
            },
        )
        get_cache = c._get_cache(
            "test",
            {
                "storage_type": ".csv",
                "storage_path": f"{self.cache_path}",
                "output_data_type": "pandas_dataframe",
            },
        )
        assert get_cache.equals(data)

    def test__get_cache_storage_type_csv_output_data_type_dask_dataframe(self):
        c = Implemented_Cacher()
        # Dask dataframe
        data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        data = dd.from_pandas(data, npartitions=2)
        c._store_cache(
            "test",
            data,
            {
                "storage_type": ".csv",
                "storage_path": f"{self.cache_path}",
                "output_data_type": "dask_dataframe",
            },
        )
        get_cache = c._get_cache(
            "test",
            {
                "storage_type": ".csv",
                "storage_path": f"{self.cache_path}",
                "output_data_type": "dask_dataframe",
            },
        )
        assert get_cache.compute().reset_index(drop=True).equals(data.compute())

    def test__get_cache_storage_type_csv_output_data_type_polars_dataframe(self):
        c = Implemented_Cacher()
        # Polars dataframe
        data = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        c._store_cache(
            "test",
            data,
            {
                "storage_type": ".csv",
                "storage_path": f"{self.cache_path}",
                "output_data_type": "polars_dataframe",
            },
        )
        get_cache = c._get_cache(
            "test",
            {
                "storage_type": ".csv",
                "storage_path": f"{self.cache_path}",
                "output_data_type": "polars_dataframe",
            },
        )
        assert data.equals(get_cache)

    def test__get_cache_storage_type_csv_output_data_type_unsupported(self):
        c = Implemented_Cacher()
        with pytest.raises(ValueError):
            c._get_cache(
                "test",
                {
                    "storage_type": ".csv",
                    "storage_path": f"{self.cache_path}",
                    "output_data_type": "new_type",
                },
            )

    # storage type .npy_stack
    def test__get_cache_storage_type_npy_stack_output_data_type_dask_array(self):
        c = Implemented_Cacher()
        # Dask array
        data = da.ones((1000, 1000), chunks=(100, 100))
        c._store_cache(
            "test",
            data,
            {
                "storage_type": ".npy_stack",
                "storage_path": f"{self.cache_path}",
                "output_data_type": "dask_array",
            },
        )
        get_cache = c._get_cache(
            "test",
            {
                "storage_type": ".npy_stack",
                "storage_path": f"{self.cache_path}",
                "output_data_type": "dask_array",
            },
        )
        assert get_cache.compute().all() == data.compute().all()

    def test__get_cache_storage_type_npy_stack_output_data_type_unsupported(self):
        c = Implemented_Cacher()
        with pytest.raises(ValueError):
            c._get_cache(
                "test",
                {
                    "storage_type": ".npy_stack",
                    "storage_path": f"{self.cache_path}",
                    "output_data_type": "numpy_array",
                },
            )

    def test__get_cache_storage_type_unsupported(self):
        c = Implemented_Cacher()
        with pytest.raises(ValueError):
            c._get_cache(
                "test",
                {
                    "storage_type": ".new_type",
                    "storage_path": f"{self.cache_path}",
                    "output_data_type": "numpy_array",
                },
            )

    # storage type .pkl
    def test__get_cache_storage_type_pkl_output_data_type_pandas_dataframe(self):
        c = Implemented_Cacher()
        # Pandas dataframe
        data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        c._store_cache(
            "test",
            data,
            {
                "storage_type": ".pkl",
                "storage_path": f"{self.cache_path}",
                "output_data_type": "pandas_dataframe",
            },
        )
        get_cache = c._get_cache(
            "test",
            {
                "storage_type": ".pkl",
                "storage_path": f"{self.cache_path}",
                "output_data_type": "pandas_dataframe",
            },
        )
        assert get_cache.equals(data)

    def test__get_cache_storage_type_pkl_output_data_type_dask_dataframe(self):
        c = Implemented_Cacher()
        # Dask dataframe
        data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        data = dd.from_pandas(data, npartitions=2)
        c._store_cache(
            "test",
            data,
            {
                "storage_type": ".pkl",
                "storage_path": f"{self.cache_path}",
                "output_data_type": "dask_dataframe",
            },
        )
        get_cache = c._get_cache(
            "test",
            {
                "storage_type": ".pkl",
                "storage_path": f"{self.cache_path}",
                "output_data_type": "dask_dataframe",
            },
        )
        assert get_cache.compute().reset_index(drop=True).equals(data.compute())
