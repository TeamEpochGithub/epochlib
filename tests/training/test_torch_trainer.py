import copy
import functools

import time
from dataclasses import dataclass

from epochalyst.training._custom_data_parallel import _CustomDataParallel

from epochalyst.training.torch_trainer import custom_collate
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch

from epochalyst.training.torch_trainer import TorchTrainer
from tests.constants import TEMP_DIR


class TestTorchTrainer:
    simple_model = torch.nn.Linear(1, 1)
    optimizer = functools.partial(torch.optim.SGD, lr=0.01)
    scheduler = functools.partial(torch.optim.lr_scheduler.StepLR, step_size=1)

    class ImplementedTorchTrainer(TorchTrainer):
        def __post_init__(self):
            self.n_folds = 1
            self.model_name = "ImplementedTorchTrainer"
            super().__post_init__()

        def log_to_terminal(self, message: str) -> None:
            print(message)

        def log_to_debug(self, message: str) -> None:
            pass

    @dataclass
    class FullyImplementedTorchTrainer(TorchTrainer):
        def __post_init__(self):
            self.n_folds = 1
            self.model_name = "FullyImplementedTorchTrainer"
            self.trained_models_directory = TEMP_DIR / "tm"
            super().__post_init__()

            self.external_logs = []

        def log_to_terminal(self, message: str) -> None:
            print(message)

        def log_to_debug(self, message: str) -> None:
            pass

        def external_define_metric(self, metric: str, metric_type: str) -> None:
            pass

        def log_to_external(self, message: dict[str, Any], **kwargs: Any) -> None:
            self.external_logs.append((message, kwargs))

        def log_to_warning(self, message: str) -> None:
            pass

    @pytest.fixture(autouse=True)
    def run_always(self, setup_temp_dir):
        pass

    def test_init_no_args(self):
        with pytest.raises(TypeError):
            TorchTrainer(n_folds=1)

    def test_init_none_args(self):
        with pytest.raises(TypeError):
            TorchTrainer(
                model=None,
                criterion=None,
                optimizer=None,
                device=None,
                n_folds=1,
            )

    def test_init_not_implemented(self):
        with pytest.raises(NotImplementedError):
            TorchTrainer(
                model=self.simple_model,
                criterion=torch.nn.MSELoss(),
                optimizer=self.optimizer,
                n_folds=0,
                model_name="Simple",
            ).external_define_metric("metric", "type")

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
        x = np.random.rand(10, 1)
        y = np.random.rand(10)
        train_indices = [0, 1, 2, 3, 4, 5, 6, 7]
        validation_indices = [8, 9]
        train_dataset, validation_dataset = tt.create_datasets(
            x, y, train_indices, validation_indices
        )
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        # Concatenate the datasets
        dataset = tt._concat_datasets(
            train_dataset, validation_dataset, train_indices, validation_indices
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
        x = np.random.rand(10, 1)
        y = np.random.rand(10)
        train_indices = [0, 1, 2, 4, 5, 7, 8, 9]
        validation_indices = [3, 6]
        train_dataset, validation_dataset = tt.create_datasets(
            x, y, train_indices, validation_indices
        )
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        # Concatenate the datasets
        dataset = tt._concat_datasets(
            train_dataset, validation_dataset, train_indices, validation_indices
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

    def test_train_no_train_indices(self):
        tt = self.ImplementedTorchTrainer(
            model=self.simple_model,
            criterion=torch.nn.MSELoss(),
            optimizer=self.optimizer,
        )
        with pytest.raises(ValueError):
            tt.train(np.random.rand(10, 1), np.random.rand(10))

    def test_train_no_test_indices(self):
        tt = self.ImplementedTorchTrainer(
            model=self.simple_model,
            criterion=torch.nn.MSELoss(),
            optimizer=self.optimizer,
        )
        with pytest.raises(ValueError):
            tt.train(
                np.random.rand(10, 1),
                np.random.rand(10),
                train_indices=[0, 1, 2, 3, 4, 5, 6, 7],
            )

    def test_train(self):
        tt = self.FullyImplementedTorchTrainer(
            model=self.simple_model,
            criterion=torch.nn.MSELoss(),
            optimizer=self.optimizer,
        )

        x = np.random.rand(10, 1)
        y = np.random.rand(10)
        tt.train(x, y, train_indices=[0, 1, 2, 3, 4, 5, 6, 7], validation_indices=[8, 9])

    def test_train_trained(self):
        tt = self.FullyImplementedTorchTrainer(
            model=self.simple_model,
            criterion=torch.nn.MSELoss(),
            optimizer=self.optimizer,
        )

        x = np.random.rand(10, 1)
        y = np.random.rand(10)
        tt.train(x, y, train_indices=[0, 1, 2, 3, 4, 5, 6, 7], validation_indices=[8, 9])
        tt.train(x, y, train_indices=[0, 1, 2, 3, 4, 5, 6, 7], validation_indices=[8, 9])

    def test_train_full(self):
        tt = self.FullyImplementedTorchTrainer(
            model=self.simple_model,
            criterion=torch.nn.MSELoss(),
            optimizer=self.optimizer,
        )
        tt.n_folds = 0

        x = np.random.rand(10, 1)
        y = np.random.rand(10)

        tt.train(x, y, train_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], validation_indices=[])
        tt.predict(x)

    def test_early_stopping(self):
        tt = self.FullyImplementedTorchTrainer(
            model=torch.nn.Sequential(
                torch.nn.Linear(1, 1),
                torch.nn.Sigmoid()),
            criterion=torch.nn.MSELoss(),
            optimizer=self.optimizer,
            patience=5,
        )

        tt.save_model_to_disk = False

        # make data on which it is guaranteed to overfit,
        # since predictions with sigmoid will be between 0 and 1
        # and train and test targets are opposite directions
        x = np.zeros((10, 1))
        y = np.array(5 * [0] + 5 * [1])
        tt.train(x, y, train_indices=[0, 1, 2, 3, 4], validation_indices=[5, 6, 7, 8, 9])

        # Check if early stopping was triggered
        assert tt.early_stopping_counter == 5

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

        x = np.random.rand(10, 1)
        y = np.random.rand(10)
        tt.train(
            x, y, train_indices=[0, 1, 2, 3, 4, 5, 6, 7], validation_indices=[8, 9], fold=0
        )
        tt.predict(x)

    def test_predict_3fold(self):
        tt = self.FullyImplementedTorchTrainer(
            model=self.simple_model,
            criterion=torch.nn.MSELoss(),
            optimizer=self.optimizer,
        )

        tt.n_folds = 3

        x = np.random.rand(10, 1)
        y = np.random.rand(10)
        tt.train(
            x, y, train_indices=[0, 1, 2, 3, 4, 5, 6, 7], validation_indices=[8, 9], fold=0
        )
        tt.train(
            x, y, train_indices=[0, 1, 2, 3, 4, 5, 6, 7], validation_indices=[8, 9], fold=1
        )
        tt.train(
            x, y, train_indices=[0, 1, 2, 3, 4, 5, 6, 7], validation_indices=[8, 9], fold=2
        )
        tt.predict(x)

    def test_predict_train_full(self):
        tt = self.FullyImplementedTorchTrainer(
            model=self.simple_model,
            criterion=torch.nn.MSELoss(),
            optimizer=self.optimizer,
        )

        tt.n_folds = 0

        x = np.random.rand(10, 1)
        y = np.random.rand(10)
        tt.train(x, y, train_indices=[0, 1, 2, 3, 4, 5, 6, 7], validation_indices=[])
        tt.predict(x)

    def test_predict2(self):
        tt = self.FullyImplementedTorchTrainer(
            model=self.simple_model,
            criterion=torch.nn.MSELoss(),
            optimizer=self.optimizer,
        )

        x = np.random.rand(10, 1)
        y = np.random.rand(10)
        tt.train(
            x,
            y,
            train_indices=np.array([0, 1, 2, 3, 4, 5, 6, 7]),
            validation_indices=np.array([8, 9]),
            fold=0,
        )
        tt.predict(x)

    def test_predict_all(self):
        tt = self.FullyImplementedTorchTrainer(
            model=self.simple_model,
            criterion=torch.nn.MSELoss(),
            optimizer=self.optimizer,
            to_predict="all",
        )

        x = np.random.rand(10, 1)
        y = np.random.rand(10)
        train_preds = tt.train(
            x,
            y,
            train_indices=np.array([0, 1, 2, 3, 4, 5, 6, 7]),
            validation_indices=np.array([8, 9]),
            fold=0,
        )
        preds = tt.predict(x)
        assert len(train_preds[0]) == 10
        assert len(preds) == 10

    def test_predict_2d(self):
        tt = self.FullyImplementedTorchTrainer(
            model=torch.nn.Linear(2, 2),
            criterion=torch.nn.MSELoss(),
            optimizer=self.optimizer,
            to_predict="all",
        )

        x = np.random.rand(10, 2)
        y = np.random.rand(10, 2)
        train_preds = tt.train(
            x,
            y,
            train_indices=np.array([0, 1, 2, 3, 4, 5, 6, 7]),
            validation_indices=np.array([8, 9]),
            fold=0,
        )
        preds = tt.predict(x)
        assert len(train_preds[0]) == 10
        assert preds.shape == (10, 2)
        assert len(preds) == 10

    def test_predict_partial(self):
        tt = self.FullyImplementedTorchTrainer(
            model=self.simple_model,
            criterion=torch.nn.MSELoss(),
            optimizer=self.optimizer,
            to_predict="validation",
        )

        x = np.random.rand(10, 1)
        y = np.random.rand(10)
        train_preds = tt.train(
            x,
            y,
            train_indices=np.array([0, 1, 2, 3, 4, 5, 6, 7]),
            validation_indices=np.array([8, 9]),
            fold=0,
        )
        preds = tt.predict(x)
        assert len(train_preds[0]) == 2
        assert len(preds) == 10

    def test_predict_none(self):
        tt = self.FullyImplementedTorchTrainer(
            model=self.simple_model,
            criterion=torch.nn.MSELoss(),
            optimizer=self.optimizer,
            to_predict="none",
        )

        x = np.random.rand(10, 1)
        y = np.random.rand(10)
        train_preds = tt.train(
            x,
            y,
            train_indices=np.array([0, 1, 2, 3, 4, 5, 6, 7]),
            validation_indices=np.array([8, 9]),
            fold=0,
        )
        preds = tt.predict(x)
        assert len(train_preds[0]) == 10
        assert len(preds) == 10

    def test_predict_no_model_trained(self):
        tt = self.FullyImplementedTorchTrainer(
            model=self.simple_model,
            criterion=torch.nn.MSELoss(),
            optimizer=self.optimizer,
        )
        with pytest.raises(FileNotFoundError):
            tt.predict(np.random.rand(10, 1))

    # Test with scheduler
    def test_train_with_scheduler(self):
        tt = self.FullyImplementedTorchTrainer(
            model=self.simple_model,
            criterion=torch.nn.MSELoss(),
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )

        x = np.random.rand(10, 1)
        y = np.random.rand(10)
        tt.train(x, y, train_indices=[0, 1, 2, 3, 4, 5, 6, 7], validation_indices=[8, 9])

    # Test 1 gpu training
    def test_train_one_gpu(self):
        with patch("torch.cuda.device_count", return_value=1):
            tt = self.FullyImplementedTorchTrainer(
                model=self.simple_model,
                criterion=torch.nn.MSELoss(),
                optimizer=self.optimizer,
            )

            x = np.random.rand(10, 1)
            y = np.random.rand(10)
            tt.train(x, y, train_indices=[0, 1, 2, 3, 4, 5, 6, 7], validation_indices=[8, 9])

    def test_train_one_gpu_saved(self):
        with patch("torch.cuda.device_count", return_value=1):
            tt = self.FullyImplementedTorchTrainer(
                model=self.simple_model,
                criterion=torch.nn.MSELoss(),
                optimizer=self.optimizer,
            )

            x = np.random.rand(10, 1)
            y = np.random.rand(10)
            tt.train(x, y, train_indices=[0, 1, 2, 3, 4, 5, 6, 7], validation_indices=[8, 9])
            tt.train(x, y, train_indices=[0, 1, 2, 3, 4, 5, 6, 7], validation_indices=[8, 9])

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

            x = np.random.rand(10, 1)
            y = np.random.rand(10)
            tt.train(x, y, train_indices=[0, 1, 2, 3, 4, 5, 6, 7], validation_indices=[8, 9])
            tt.train(x, y, train_indices=[0, 1, 2, 3, 4, 5, 6, 7], validation_indices=[8, 9])

    def test_early_stopping_no_patience(self):
        tt = self.FullyImplementedTorchTrainer(
            model=self.simple_model,
            criterion=torch.nn.MSELoss(),
            optimizer=self.optimizer,
            patience=-1,
        )

        orig_state_dict = copy.deepcopy(tt.model.state_dict)

        tt._early_stopping()

        # Lowest val loss should still be -inf
        assert np.isinf(tt.lowest_val_loss)

        # Early stopping counter should not exist
        assert not hasattr(tt, "early_stopping_counter")

        x = np.random.rand(10, 1)
        y = np.random.rand(10)
        tt.train(x, y, train_indices=[0, 1, 2, 3, 4, 5, 6, 7], validation_indices=[8, 9])
        # Assert that no best model exists
        assert tt.best_model_state_dict == {}
        # Assert self.model still exists
        assert tt.model.state_dict != {}
        # Assert model chnages after training
        assert tt.model.state_dict != orig_state_dict

    def test_custom_collate(self):
        x = torch.rand(10, 1)
        y = torch.rand(10)
        collate_x, collate_y = custom_collate((x, y))
        assert torch.all(collate_x == x) and torch.all(collate_y == y)

    def test_checkpointing(self):
        tt = self.FullyImplementedTorchTrainer(
            model=self.simple_model,
            criterion=torch.nn.MSELoss(),
            optimizer=self.optimizer,
            patience=-1,
            checkpointing_enabled=True,
            checkpointing_keep_every=1,
            checkpointing_resume_if_exists=True,
        )

        x = np.random.rand(10, 1)
        y = np.random.rand(10)

        # Train once
        time_temp = time.time()
        tt.train(x, y, train_indices=[0, 1, 2, 3, 4, 5, 6, 7], validation_indices=[8, 9])
        spent_time_first_run = time.time() - time_temp

        # Check if checkpoints exist
        saved_checkpoints = list(tt.trained_models_directory.glob("*_checkpoint_*.pt"))
        assert len(saved_checkpoints) == tt.epochs

        # Remove model and all but the 2nd to last checkpoint
        start_epoch = tt.epochs - 2
        epochs = [
            int(checkpoint.stem.split("_")[-1]) for checkpoint in saved_checkpoints
        ]
        checkpoint_to_keep = saved_checkpoints[epochs.index(start_epoch)]
        print(checkpoint_to_keep)
        for file in tt.trained_models_directory.glob("*.pt"):
            if file != checkpoint_to_keep:
                file.unlink()

        # Train again
        time_temp = time.time()
        tt.train(x, y, train_indices=[0, 1, 2, 3, 4, 5, 6, 7], validation_indices=[8, 9])
        spent_time_second_run = time.time() - time_temp

        # Check if checkpoints exist
        saved_checkpoints = list(tt.trained_models_directory.glob("*_checkpoint_*.pt"))
        assert len(saved_checkpoints) == tt.epochs - start_epoch

        # Check if training time was signficicantly less the second time
        assert spent_time_second_run < (spent_time_first_run / 2)

    def test_onnx(self):
        tt = self.FullyImplementedTorchTrainer(
            model=self.simple_model,
            criterion=torch.nn.MSELoss(),
            optimizer=self.optimizer,
            patience=-1,
        )
        tt.device = torch.device("cpu")
        if isinstance(tt.model, _CustomDataParallel):
            tt.model = tt.model.module
        tt.model.to('cpu')
        x = np.random.rand(10, 1)
        y = np.random.rand(10)
        _ = tt.train(
            x,
            y,
            train_indices=np.array([0, 1, 2, 3, 4, 5, 6, 7]),
            validation_indices=np.array([8, 9]),
            fold=0,
        )
        onnx_preds = tt.predict(x, None, **{'compile_method': 'ONNX'})
        preds = tt.predict(x)
        assert np.allclose(onnx_preds, preds[:, np.newaxis])

    def test_openvino(self):
        tt = self.FullyImplementedTorchTrainer(
            model=self.simple_model,
            criterion=torch.nn.MSELoss(),
            optimizer=self.optimizer,
            patience=-1,
        )
        tt.device = torch.device("cpu")
        if isinstance(tt.model, _CustomDataParallel):
            tt.model = tt.model.module
        tt.model.to('cpu')
        x = np.random.rand(10, 1)
        y = np.random.rand(10)
        _ = tt.train(
            x,
            y,
            train_indices=np.array([0, 1, 2, 3, 4, 5, 6, 7]),
            validation_indices=np.array([8, 9]),
            fold=0,
        )
        openvino_preds = tt.predict(x, None, **{'compile_method': 'Openvino'})
        preds = tt.predict(x)
        assert np.allclose(openvino_preds, preds[:, np.newaxis])

    def test_log_external_train(self):
        tt = self.FullyImplementedTorchTrainer(
            model=self.simple_model,
            criterion=torch.nn.MSELoss(),
            optimizer=self.optimizer,
            epochs=1,
        )
        # train for 1 epoch
        tt.epochs = 1
        x = np.random.rand(10, 1)
        y = np.random.rand(10)
        tt.train(x, y, train_indices=[0, 1, 2, 3, 4, 5, 6, 7], validation_indices=[8, 9])

        # Check if logs are stored
        assert len(tt.external_logs) == 4

        # train loss
        assert 'Training/Train Loss' in tt.external_logs[0][0]
        assert tt.external_logs[0][0].get('epoch') == 0

        # validation loss
        assert 'Validation/Validation Loss' in tt.external_logs[1][0]
        assert tt.external_logs[1][0].get('epoch') == 0

        # loss table
        table = tt.external_logs[2][0]
        assert table.get('type') == 'wandb_plot'
        assert table.get('data', dict()).get('keys') == ['Train', 'Validation']
        assert table.get('data', dict()).get('title') == 'Training/Loss'

        # early stopping epochs
        assert tt.external_logs[3][0].get('Epochs') == 1

    def test_log_external_prefix_postfix(self):
        tt = self.FullyImplementedTorchTrainer(
            model=self.simple_model,
            criterion=torch.nn.MSELoss(),
            optimizer=self.optimizer,
            epochs=1,
            logging_prefix='prefix',
            logging_postfix='postfix'
        )

        x = np.random.rand(10, 1)
        y = np.random.rand(10)
        tt.train(x, y, train_indices=[0, 1, 2, 3, 4, 5, 6, 7], validation_indices=[8, 9])

        # Check if logs are stored
        assert len(tt.external_logs) == 4

        # train loss
        assert 'prefixTraining/Train Losspostfix' in tt.external_logs[0][0]
        assert tt.external_logs[0][0].get('prefixepochpostfix') == 0

        # validation loss
        assert 'prefixValidation/Validation Losspostfix' in tt.external_logs[1][0]
        assert tt.external_logs[1][0].get('prefixepochpostfix') == 0

        # loss table
        table = tt.external_logs[2][0]
        assert table.get('type') == 'wandb_plot'
        assert table.get('data', dict()).get('keys') == ['Train', 'Validation']
        assert table.get('data', dict()).get('title') == 'prefixTraining/Losspostfix'

        # early stopping epochs
        assert tt.external_logs[3][0].get('prefixEpochspostfix') == 1

    def test_onnx_raises_error(self):
        tt = self.FullyImplementedTorchTrainer(
            model=self.simple_model,
            criterion=torch.nn.MSELoss(),
            optimizer=self.optimizer,
            patience=-1,
        )

        x = np.random.rand(10, 1)
        y = np.random.rand(10)
        _ = tt.train(
            x,
            y,
            train_indices=np.array([0, 1, 2, 3, 4, 5, 6, 7]),
            validation_indices=np.array([8, 9]),
            fold=0,
        )
        tt.device = 'gpu'
        with pytest.raises(ValueError):
            _ = tt.predict(x, None, **{'compile_method': 'ONNX'})

    def test_openvino_raises_error(self):
        tt = self.FullyImplementedTorchTrainer(
            model=self.simple_model,
            criterion=torch.nn.MSELoss(),
            optimizer=self.optimizer,
            patience=-1,
        )

        x = np.random.rand(10, 1)
        y = np.random.rand(10)
        _ = tt.train(
            x,
            y,
            train_indices=np.array([0, 1, 2, 3, 4, 5, 6, 7]),
            validation_indices=np.array([8, 9]),
            fold=0,
        )
        tt.device = 'gpu'
        with pytest.raises(ValueError):
            _ = tt.predict(x, None, **{'compile_method': 'Openvino'})
