from pathlib import Path

import pytest

from epochalyst.training import TrainingBlock

TEMP_DIR = Path("tests/temp")


class TestTrainingBlock:
    cache_path = TEMP_DIR

    def test_training_block_init(self):
        tb = TrainingBlock()
        assert tb is not None

    def test_training_block_train(self):
        with pytest.raises(NotImplementedError):
            tb = TrainingBlock()
            tb.train(1, 1)

    def test_training_block_predict(self):
        with pytest.raises(NotImplementedError):
            tb = TrainingBlock()
            tb.predict(1)

    def test_training_block_predict_implemented(self):
        class TestTrainingBlockImpl(TrainingBlock):
            def custom_train(self, x: int, y: int) -> tuple[int, int]:
                return x, y

            def custom_predict(self, x: int) -> int:
                return x

        tb = TestTrainingBlockImpl()
        assert tb.train(1, 1) == (1, 1)
        assert tb.predict(1) == 1

    def test_training_block_caching(self, setup_temp_dir):
        class TestTrainingBlockImpl(TrainingBlock):
            def custom_train(self, x: int, y: int) -> tuple[int, int]:
                return x, y

            def custom_predict(self, x: int) -> int:
                return x

            def log_to_debug(self, message: str) -> None:
                return None

            def log_to_terminal(self, message: str) -> None:
                return None

        tb = TestTrainingBlockImpl()
        cache_args = {
            "output_data_type": "numpy_array",
            "storage_type": ".npy",
            "storage_path": f"{self.cache_path}",
        }

        x, y = tb.train(1, 1, cache_args=cache_args)
        new_x, new_y = tb.train(1, 1, cache_args=cache_args)
        assert x == new_x
        assert y == new_y

        pred = tb.predict(1, cache_args=cache_args)
        new_pred = tb.predict(1, cache_args=cache_args)
        assert pred == new_pred
