import functools
from typing import Any
from unittest.mock import patch
import torch
from epochalyst.pipeline.model.training.torch_trainer import TorchTrainer
import pytest

from tests.util import remove_cache_files


class TestTorchTrainer:
    simple_model = torch.nn.Linear(1, 1)
    optimizer = functools.partial(torch.optim.SGD, lr=0.01)
    scheduler = functools.partial(torch.optim.lr_scheduler.StepLR, step_size=1)

    class ImplementedTorchTrainer(TorchTrainer):
        def log_to_terminal(self, message: str) -> None:
            print(message)

        def log_to_debug(self, message: str) -> None:
            pass

    class FullyImplementedTorchTrainer(TorchTrainer):
        def log_to_terminal(self, message: str) -> None:
            print(message)

        def log_to_debug(self, message: str) -> None:
            pass

        def external_define_metric(self, metric: str, metric_type: str) -> None:
            pass

        def log_to_external(self, message: dict[str, Any], **kwargs: Any) -> None:
            pass

        def log_to_warning(self, message: str) -> None:
            pass

    def test_init_no_args(self):
        with pytest.raises(TypeError):
            TorchTrainer()

    def test_init_none_args(self):
        with pytest.raises(TypeError):
            TorchTrainer(model=None, criterion=None, optimizer=None, device=None)

    def test_init_proper_args(self):
        with pytest.raises(NotImplementedError):
            TorchTrainer(
                model=self.simple_model,
                criterion=torch.nn.MSELoss(),
                optimizer=self.optimizer,
            )

    def test_init_proper_args_with_implemented(self):
        tt = self.ImplementedTorchTrainer(
            model=self.simple_model,
            criterion=torch.nn.MSELoss(),
            optimizer=self.optimizer,
        )
        assert tt is not None

    # Dataset concatenation
    def test__concat_datasets_in_order(self):
        tt = self.FullyImplementedTorchTrainer(
            model=self.simple_model,
            criterion=torch.nn.MSELoss(),
            optimizer=self.optimizer,
        )
        x = torch.rand(10, 1)
        y = torch.rand(10)
        train_indices = [0, 1, 2, 3, 4, 5, 6, 7]
        test_indices = [8, 9]
        train_dataset, test_dataset = tt.create_datasets(
            x, y, train_indices, test_indices
        )

        # Concatenate the datasets
        dataset = tt._concat_datasets(
            train_dataset, test_dataset, train_indices, test_indices
        )
        assert len(dataset) == 10

        # Assert values of dataset are correct
        for i in range(10):
            assert (dataset[i][0] == x[i]).all()
            assert (dataset[i][1] == y[i]).all()

    def test__concat_datasets_out_of_order(self):
        tt = self.FullyImplementedTorchTrainer(
            model=self.simple_model,
            criterion=torch.nn.MSELoss(),
            optimizer=self.optimizer,
        )
        x = torch.rand(10, 1)
        y = torch.rand(10)
        train_indices = [0, 1, 2, 4, 5, 7, 8, 9]
        test_indices = [3, 6]
        train_dataset, test_dataset = tt.create_datasets(
            x, y, train_indices, test_indices
        )

        # Concatenate the datasets
        dataset = tt._concat_datasets(
            train_dataset, test_dataset, train_indices, test_indices
        )

        # Check the length of the dataset
        assert len(dataset) == 10

        # Assert values of dataset are correct
        for i in range(10):
            assert (dataset[i][0] == x[i]).all()
            assert (dataset[i][1] == y[i]).all()

    # Training
    def test_train_no_args(self):
        tt = self.ImplementedTorchTrainer(
            model=self.simple_model,
            criterion=torch.nn.MSELoss(),
            optimizer=self.optimizer,
        )
        with pytest.raises(TypeError):
            tt.train()

    def test_train(self):
        tt = self.FullyImplementedTorchTrainer(
            model=self.simple_model,
            criterion=torch.nn.MSELoss(),
            optimizer=self.optimizer,
        )
        tt.update_model_directory("tests/cache")
        x = torch.rand(10, 1)
        y = torch.rand(10)
        tt.train(x, y, train_indices=[0, 1, 2, 3, 4, 5, 6, 7], test_indices=[8, 9])

        remove_cache_files()

    def test_train_trained(self):
        tt = self.FullyImplementedTorchTrainer(
            model=self.simple_model,
            criterion=torch.nn.MSELoss(),
            optimizer=self.optimizer,
        )
        tt.update_model_directory("tests/cache")
        x = torch.rand(10, 1)
        y = torch.rand(10)
        tt.train(x, y, train_indices=[0, 1, 2, 3, 4, 5, 6, 7], test_indices=[8, 9])
        tt.train(x, y, train_indices=[0, 1, 2, 3, 4, 5, 6, 7], test_indices=[8, 9])

        remove_cache_files()

    def test_train_full(self):
        tt = self.FullyImplementedTorchTrainer(
            model=self.simple_model,
            criterion=torch.nn.MSELoss(),
            optimizer=self.optimizer,
        )
        tt.update_model_directory("tests/cache")
        x = torch.rand(10, 1)
        y = torch.rand(10)

        tt.train(x, y, train_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], test_indices=[])

        remove_cache_files()

    def test_early_stopping(self):
        tt = self.FullyImplementedTorchTrainer(
            model=self.simple_model,
            criterion=torch.nn.MSELoss(),
            optimizer=self.optimizer,
            patience=-1,
        )
        tt.update_model_directory("tests/cache")
        x = torch.rand(10, 1)
        y = torch.rand(10)
        tt.train(x, y, train_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], test_indices=[])

        remove_cache_files()

    # Test predict
    def test_predict_no_args(self):
        tt = self.ImplementedTorchTrainer(
            model=self.simple_model,
            criterion=torch.nn.MSELoss(),
            optimizer=self.optimizer,
        )
        with pytest.raises(TypeError):
            tt.predict()

    def test_predict(self):
        tt = self.FullyImplementedTorchTrainer(
            model=self.simple_model,
            criterion=torch.nn.MSELoss(),
            optimizer=self.optimizer,
        )
        tt.update_model_directory("tests/cache")
        x = torch.rand(10, 1)
        y = torch.rand(10)
        tt.train(x, y, train_indices=[0, 1, 2, 3, 4, 5, 6, 7], test_indices=[8, 9])
        tt.predict(x)

        remove_cache_files()

    def test_predict_no_model_trained(self):
        tt = self.FullyImplementedTorchTrainer(
            model=self.simple_model,
            criterion=torch.nn.MSELoss(),
            optimizer=self.optimizer,
        )
        with pytest.raises(FileNotFoundError):
            tt.predict(torch.rand(10, 1))

    # Test with scheduler
    def test_train_with_scheduler(self):
        tt = self.FullyImplementedTorchTrainer(
            model=self.simple_model,
            criterion=torch.nn.MSELoss(),
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )
        tt.update_model_directory("tests/cache")
        x = torch.rand(10, 1)
        y = torch.rand(10)
        tt.train(x, y, train_indices=[0, 1, 2, 3, 4, 5, 6, 7], test_indices=[8, 9])

        remove_cache_files()

    # Test 1 gpu training
    def test_train_one_gpu(self):
        with patch("torch.cuda.device_count", return_value=1):
            tt = self.FullyImplementedTorchTrainer(
                model=self.simple_model,
                criterion=torch.nn.MSELoss(),
                optimizer=self.optimizer,
            )

            tt.update_model_directory("tests/cache")
            x = torch.rand(10, 1)
            y = torch.rand(10)
            tt.train(x, y, train_indices=[0, 1, 2, 3, 4, 5, 6, 7], test_indices=[8, 9])

        remove_cache_files()

    def test_train_one_gpu_saved(self):
        with patch("torch.cuda.device_count", return_value=1):
            tt = self.FullyImplementedTorchTrainer(
                model=self.simple_model,
                criterion=torch.nn.MSELoss(),
                optimizer=self.optimizer,
            )

            tt.update_model_directory("tests/cache")
            x = torch.rand(10, 1)
            y = torch.rand(10)
            tt.train(x, y, train_indices=[0, 1, 2, 3, 4, 5, 6, 7], test_indices=[8, 9])
            tt.train(x, y, train_indices=[0, 1, 2, 3, 4, 5, 6, 7], test_indices=[8, 9])

        remove_cache_files()

    def test_train_two_gpu_saved(self):
        # If test is run on a machine with 2 or more GPUs, this test will run else it will be skipped
        if torch.cuda.device_count() < 2:
            pytest.skip("Test requires 2 GPUs")

        with patch("torch.cuda.device_count", return_value=2):
            tt = self.FullyImplementedTorchTrainer(
                model=self.simple_model,
                criterion=torch.nn.MSELoss(),
                optimizer=self.optimizer,
            )

            tt.update_model_directory("tests/cache")
            x = torch.rand(10, 1)
            y = torch.rand(10)
            tt.train(x, y, train_indices=[0, 1, 2, 3, 4, 5, 6, 7], test_indices=[8, 9])
            tt.train(x, y, train_indices=[0, 1, 2, 3, 4, 5, 6, 7], test_indices=[8, 9])

        remove_cache_files()
