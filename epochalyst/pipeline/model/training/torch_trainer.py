import copy
from dataclasses import dataclass
import functools
import gc
from pathlib import Path
from typing import Annotated, Any, Callable
from annotated_types import Gt, Interval
import numpy as np
from torch import Tensor, nn
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm
from torch.utils.data import Dataset, TensorDataset, DataLoader

from epochalyst._core._pipeline._custom_data_parallel import _CustomDataParallel
from epochalyst.logging.section_separator import print_section_separator
import numpy.typing as npt

from epochalyst.pipeline.model.training.training_block import TrainingBlock


@dataclass
class TorchTrainer(TrainingBlock):
    """Abstract class for torch trainers, override necessary functions for custom implementation.

    ### Parameters:
    - `model` (nn.Module): The model to train.
    - `optimizer` (functools.partial[Optimizer]): Optimizer to use for training.
    - `criterion` (nn.Module): Criterion to use for training.
    - `scheduler` (Callable[[Optimizer], LRScheduler] | None): Scheduler to use for training.
    - `epochs` (int): Number of epochs
    - `batch_size` (int): Batch size
    - `patience` (int): Patience for early stopping
    - `test_size` (float): Relative size of the test set

    ### Methods:
    ```python
    @abstractmethod
    def log_to_terminal(self, message: str) -> None:
        # Logs to terminal if implemented

    @abstractmethod
    def log_to_debug(self, message: str) -> None:
        # Logs to debugger if implemented

    @abstractmethod
    def log_to_warning(self, message: str) -> None:
        # Logs to warning if implemented

    @abstractmethod
    def log_to_external(self, message: dict[str, Any], **kwargs: Any) -> None:
        # Logs to external site

    @abstractmethod
    def external_define_metric(self, metric: str, metric_type: str) -> None:
        # Defines an external metric

    def train(self, x: Any, y: Any, cache_args: dict[str, Any] = {}, **train_args: Any) -> tuple[Any, Any]:
        # Applies caching and calls custom_train, overridding removes caching

    def predict(self, x: Any, cache_args: dict[str, Any] = {}, **pred_args: Any) -> Any:
        # Applies caching and calls custom_predict, overridding removes caching

    def custom_train(self, x: Any, y: Any, **train_args: Any) -> tuple[Any, Any]:
        # Implements torch training. If you are going to override this method and not use any other functionality, inherit from TrainingBlock.

    def custom_predict(self, x: Any, **pred_args: Any) -> Any:
        # Implements torch prediction. If you are going to override this method and not use any other functionality, inherit from TrainingBlock.

    def predict_on_loader(loader: DataLoader[tuple[Tensor, ...]]) -> npt.NDArray[np.float32]:
        # Predict using a dataloader.

    def create_datasets(x: npt.NDArray[np.float32], y: npt.NDArray[np.float32], train_indices: list[int], test_indices: list[int], cache_size: int = -1) -> tuple[Dataset[tuple[Tensor, ...]], Dataset[tuple[Tensor, ...]]]:
        # Create the datasets for training and validation.

    def create_prediction_dataset(x: npt.NDArray[np.float32]) -> Dataset[tuple[Tensor, ...]]:
        # Create the prediction dataset.

    def create_dataloaders(train_dataset: Dataset[tuple[Tensor, ...]], test_dataset: Dataset[tuple[Tensor, ...]]) -> tuple[DataLoader[tuple[Tensor, ...]], DataLoader[tuple[Tensor, ...]]]:
        # Create the dataloaders for training and validation.

    def update_model_directory(model_directory: str) -> None:
        # Update the model directory for caching (default: tm).
    ```

    ### Usage:
    ```python
    from epochalyst.pipeline.model.training.torch_trainer import TorchTrainer
    from torch import nn
    from torch.optim import Adam
    from torch.optim.lr_scheduler import StepLR
    from torch.nn import MSELoss

    class MyTorchTrainer(TorchTrainer):

        def log_to_terminal(self, message: str) -> None:

        ....

    model = nn.Sequential(nn.Linear(1, 1))
    optimizer = functools.partial(Adam, lr=0.01)
    criterion = MSELoss()
    scheduler = functools.partial(StepLR, step_size=1, gamma=0.1)
    epochs = 10
    batch_size = 32
    patience = 5
    test_size = 0.2

    trainer = MyTorchTrainer(model=model, optimizer=optimizer, criterion=criterion, scheduler=scheduler, epochs=epochs, batch_size=batch_size, patience=patience, test_size=test_size)

    x, y = trainer.train(x, y)
    x = trainer.predict(x)
    ```
    """

    model: nn.Module
    optimizer: functools.partial[Optimizer]
    criterion: nn.Module
    scheduler: Callable[[Optimizer], LRScheduler] | None = None
    epochs: Annotated[int, Gt(0)] = 10
    batch_size: Annotated[int, Gt(0)] = 32
    patience: Annotated[int, Gt(0)] = 5
    test_size: Annotated[float, Interval(ge=0, le=1)] = 0.2  # Hashing purposes

    def __post_init__(self) -> None:
        """Post init method for the TorchTrainer class."""

        self.save_model_to_disk = True
        self.model_directory = "tm"
        self.best_model_state_dict: dict[Any, Any] = {}

        # Set optimizer
        self.initialized_optimizer = self.optimizer(self.model.parameters())

        # Set scheduler
        self.initialized_scheduler: LRScheduler | None
        if self.scheduler is not None:
            self.initialized_scheduler = self.scheduler(self.initialized_optimizer)
        else:
            self.initialized_scheduler = None

        # Set the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_to_terminal(f"Setting device: {self.device}")

        # If multiple GPUs are available, distribute batch size over the GPUs
        if torch.cuda.device_count() > 1:
            self.log_to_terminal(f"Using {torch.cuda.device_count()} GPUs")
            self.model = _CustomDataParallel(self.model)

        self.model.to(self.device)

        # Early stopping
        self.last_val_loss = np.inf
        self.lowest_val_loss = np.inf

        super().__post_init__()

    def custom_train(
        self,
        x: npt.NDArray[np.float32],
        y: npt.NDArray[np.float32],
        **train_args: Any,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Train the model.

        :param x: The input to the system.
        :param y: The expected output of the system.

        Keyword Arguments:
        :param train_indices: The indices to train on.
        :param test_indices: The indices to test on.
        :param cache_size: The cache size.
        :param save_model: Whether to save the model.
        :return: The input and output of the system.
        """
        train_indices = train_args.get("train_indices")
        if train_indices is None:
            raise ValueError("train_indices not provided")
        test_indices = train_args.get("test_indices")
        if test_indices is None:
            raise ValueError("test_indices not provided")
        cache_size = train_args.get("cache_size", -1)
        save_model = train_args.get("save_model", True)

        self.save_model_to_disk = save_model
        if self._model_exists():
            self.log_to_terminal(
                f"Model exists in {self.model_directory}/{self.get_hash()}.pt, loading model"
            )
            self._load_model()
            return self.custom_predict(x), y

        self.log_to_terminal(f"Training model: {self.model.__class__.__name__}")
        self.log_to_debug(f"Training model: {self.model.__class__.__name__}")

        # Create datasets
        train_dataset, test_dataset = self.create_datasets(
            x, y, train_indices, test_indices, cache_size=cache_size
        )

        # Create dataloaders
        train_loader, test_loader = self.create_dataloaders(train_dataset, test_dataset)

        # Train the model
        self.log_to_terminal(f"Training model for {self.epochs} epochs")

        train_losses: list[float] = []
        val_losses: list[float] = []

        self.external_define_metric("Training/Train Loss", "min")
        self.external_define_metric("Validation/Validation Loss", "min")

        self.lowest_val_loss = np.inf
        if len(test_loader) == 0:
            self.log_to_warning(
                f"Doing train full, model will be trained for {self.epochs} epochs"
            )

        self._training_loop(train_loader, test_loader, train_losses, val_losses)

        self.log_to_terminal(
            f"Done training the model: {self.model.__class__.__name__}"
        )

        # Revert to the best model
        if self.best_model_state_dict:
            self.log_to_terminal(
                f"Reverting to model with best validation loss {self.lowest_val_loss}"
            )
            self.model.load_state_dict(self.best_model_state_dict)

        if save_model:
            self._save_model()

        # Return the predictions
        concat_dataset = self._concat_datasets(
            train_dataset, test_dataset, train_indices, test_indices
        )

        pred_dataloader = DataLoader(
            concat_dataset, batch_size=self.batch_size, shuffle=False
        )

        return self.predict_on_loader(pred_dataloader), y

    def custom_predict(
        self, x: npt.NDArray[np.float32], **pred_args: Any
    ) -> npt.NDArray[np.float32]:
        """Predict on the test data

        :param x: The input to the system.
        :return: The output of the system.
        """
        self._load_model()

        print_section_separator(f"Predicting model: {self.model.__class__.__name__}")
        self.log_to_debug(f"Predicting model: {self.model.__class__.__name__}")

        # Create dataset
        pred_dataset = self.create_prediction_dataset(x)
        pred_dataloader = DataLoader(
            pred_dataset, batch_size=self.batch_size, shuffle=False
        )

        # Predict
        return self.predict_on_loader(pred_dataloader)

    def predict_on_loader(
        self, loader: DataLoader[tuple[Tensor, ...]]
    ) -> npt.NDArray[np.float32]:
        """Predict on the loader.

        :param loader: The loader to predict on.
        :return: The predictions.
        """
        self.log_to_terminal("Predicting on the test data")
        self.model.eval()
        predictions = []
        with torch.no_grad(), tqdm(loader, unit="batch", disable=False) as tepoch:
            for data in tepoch:
                X_batch = data[0].to(self.device).float()

                y_pred = self.model(X_batch).cpu().numpy()
                predictions.extend(y_pred)

        self.log_to_terminal("Done predicting")
        return np.array(predictions)

    def create_datasets(
        self,
        x: npt.NDArray[np.float32],
        y: npt.NDArray[np.float32],
        train_indices: list[int],
        test_indices: list[int],
        cache_size: int = -1,
    ) -> tuple[Dataset[tuple[Tensor, ...]], Dataset[tuple[Tensor, ...]]]:
        """Create the datasets for training and validation.

        :param x: The input data.
        :param y: The target variable.
        :param train_indices: The indices to train on.
        :param test_indices: The indices to test on.
        :return: The training and validation datasets.
        """
        train_dataset = TensorDataset(
            torch.tensor(x[train_indices]), torch.tensor(y[train_indices])
        )
        test_dataset = TensorDataset(
            torch.tensor(x[test_indices]), torch.tensor(y[test_indices])
        )

        return train_dataset, test_dataset

    def create_prediction_dataset(
        self, x: npt.NDArray[np.float32]
    ) -> Dataset[tuple[Tensor, ...]]:
        """Create the prediction dataset.

        :param x: The input data.
        :return: The prediction dataset.
        """
        return TensorDataset(torch.tensor(x))

    def create_dataloaders(
        self,
        train_dataset: Dataset[tuple[Tensor, ...]],
        test_dataset: Dataset[tuple[Tensor, ...]],
    ) -> tuple[DataLoader[tuple[Tensor, ...]], DataLoader[tuple[Tensor, ...]]]:
        """Create the dataloaders for training and validation.

        :param train_dataset: The training dataset.
        :param test_dataset: The validation dataset.
        :return: The training and validation dataloaders.
        """
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )
        return train_loader, test_loader

    def update_model_directory(self, model_directory: str) -> None:
        """Update the model directory.

        :param model_directory: The model directory.
        """
        self.model_directory = model_directory

    def _training_loop(
        self,
        train_loader: DataLoader[tuple[Tensor, ...]],
        test_loader: DataLoader[tuple[Tensor, ...]],
        train_losses: list[float],
        val_losses: list[float],
    ) -> None:
        """Training loop for the model.

        :param train_loader: Dataloader for the testing data.
        :param test_loader: Dataloader for the training data. (can be empty)
        :param train_losses: List of train losses.
        :param val_losses: List of validation losses.
        """
        for epoch in range(self.epochs):
            # Train using train_loader
            train_loss = self._train_one_epoch(train_loader, epoch)
            self.log_to_debug(f"Epoch {epoch} Train Loss: {train_loss}")
            train_losses.append(train_loss)

            # Log train loss
            self.log_to_external(
                message={"Training/Train Loss": train_losses[-1]}, step=epoch + 1
            )

            # Compute validation loss
            if len(test_loader) > 0:
                self.last_val_loss = self._val_one_epoch(
                    test_loader, desc=f"Epoch {epoch} Valid"
                )
                self.log_to_debug(f"Epoch {epoch} Valid Loss: {self.last_val_loss}")
                val_losses.append(self.last_val_loss)

                # Log validation loss and plot train/val loss against each other
                self.log_to_external(
                    message={"Validation/Validation Loss": val_losses[-1]},
                    step=epoch + 1,
                )

                # TODO(Jasper): How to log this without wandb?
                # wandb.log(
                #         {
                #             "Training/Loss": wandb.plot.line_series(
                #                 xs=range(epoch + 1),
                #                 ys=[train_losses, val_losses],
                #                 keys=["Train", "Validation"],
                #                 title="Training/Loss",
                #                 xname="Epoch",
                #             ),
                #         },
                #     )

                # Early stopping
                if self._early_stopping():
                    self.log_to_external(
                        message={"Epochs": (epoch + 1) - self.patience}
                    )
                    break

            # Log the trained epochs to wandb if we finished training
            self.log_to_external(message={"Epochs": epoch + 1})

    def _train_one_epoch(
        self, dataloader: DataLoader[tuple[Tensor, ...]], epoch: int
    ) -> float:
        """Train the model for one epoch.

        :param dataloader: Dataloader for the training data.
        :param epoch: Epoch number.
        :return: Average loss for the epoch.
        """
        losses = []
        self.model.train()
        pbar = tqdm(dataloader, unit="batch", desc=f"Epoch {epoch} Train")
        for batch in pbar:
            X_batch, y_batch = batch
            X_batch = X_batch.to(self.device).float()
            y_batch = y_batch.to(self.device).float()

            # Forward pass
            y_pred = self.model(X_batch).squeeze(1)
            loss = self.criterion(y_pred, y_batch)

            # Backward pass
            self.initialized_optimizer.zero_grad()
            loss.backward()
            self.initialized_optimizer.step()

            # Print tqdm
            losses.append(loss.item())
            pbar.set_postfix(loss=sum(losses) / len(losses))

        # Step the scheduler
        if self.initialized_scheduler is not None:
            self.initialized_scheduler.step(epoch=epoch)

        # Remove the cuda cache
        torch.cuda.empty_cache()
        gc.collect()

        return sum(losses) / len(losses)

    def _val_one_epoch(
        self, dataloader: DataLoader[tuple[Tensor, ...]], desc: str
    ) -> float:
        """Compute validation loss of the model for one epoch.

        :param dataloader: Dataloader for the testing data.
        :param desc: Description for the tqdm progress bar.
        :return: Average loss for the epoch.
        """
        losses = []
        self.model.eval()
        pbar = tqdm(dataloader, unit="batch")
        with torch.no_grad():
            for batch in pbar:
                X_batch, y_batch = batch
                X_batch = X_batch.to(self.device).float()
                y_batch = y_batch.to(self.device).float()

                # Forward pass
                y_pred = self.model(X_batch).squeeze(1)
                loss = self.criterion(y_pred, y_batch)

                # Print losses
                losses.append(loss.item())
                pbar.set_description(desc=desc)
                pbar.set_postfix(loss=sum(losses) / len(losses))
        return sum(losses) / len(losses)

    def _save_model(self) -> None:
        """Save the model in the model_directory folder."""
        self.log_to_terminal(
            f"Saving model to {self.model_directory}/{self.get_hash()}.pt"
        )
        path = Path(self.model_directory)
        if not Path.exists(path):
            Path.mkdir(path)

        torch.save(self.model, f"{self.model_directory}/{self.get_hash()}.pt")
        self.log_to_terminal(
            f"Model saved to {self.model_directory}/{self.get_hash()}.pt"
        )

    def _load_model(self) -> None:
        """Load the model from the model_directory folder."""

        # Check if the model exists
        if not Path(f"{self.model_directory}/{self.get_hash()}.pt").exists():
            raise FileNotFoundError(
                f"Model not found in {self.model_directory}/{self.get_hash()}.pt"
            )

        # Load model
        self.log_to_terminal(
            f"Loading model from {self.model_directory}/{self.get_hash()}.pt"
        )
        checkpoint = torch.load(f"{self.model_directory}/{self.get_hash()}.pt")

        # Load the weights from the checkpoint
        if isinstance(checkpoint, nn.DataParallel):
            model = checkpoint.module
        else:
            model = checkpoint

        # Set the current model to the loaded model
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(model.state_dict())
        else:
            self.model.load_state_dict(model.state_dict())

        self.log_to_terminal(
            f"Model loaded from {self.model_directory}/{self.get_hash()}.pt"
        )

    def _model_exists(self) -> bool:
        """Check if the model exists in the model_directory folder."""
        return (
            Path(f"{self.model_directory}/{self.get_hash()}.pt").exists()
            and self.save_model_to_disk
        )

    def _early_stopping(self) -> bool:
        """Check if early stopping should be performed.

        :return: Whether to perform early stopping.
        """

        # Store the best model so far based on validation loss
        if self.last_val_loss < self.lowest_val_loss:
            self.lowest_val_loss = self.last_val_loss
            self.best_model_state_dict = copy.deepcopy(self.model.state_dict())
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
            if self.early_stopping_counter >= self.patience:
                self.log_to_terminal(
                    f"Early stopping after {self.early_stopping_counter} epochs"
                )
                return True
        return False

    def _concat_datasets(
        self,
        train_dataset: Dataset[tuple[Tensor, ...]],
        test_dataset: Dataset[tuple[Tensor, ...]],
        train_indices: list[int],
        test_indices: list[int],
    ) -> Dataset[tuple[Tensor, ...]]:
        """
        Concatenate the training and test datasets according to original order specified by train_indices and test_indices.

        :param train_dataset: The training dataset.
        :param test_dataset: The test dataset.
        :param train_indices: The indices for the training data.
        :param test_indices: The indices for the test data.
        :return: A new dataset containing the concatenated data in the original order.
        """

        # Combine the indices and sort them alongside their corresponding dataset identifier ('train' or 'test')
        combined_indices = train_indices + test_indices
        dataset_labels = ["train"] * len(train_indices) + ["test"] * len(test_indices)
        sorted_combined = sorted(
            zip(combined_indices, dataset_labels), key=lambda x: x[0]
        )

        # Create a new list to hold the concatenated dataset
        concatenated_dataset = []

        # Iterate over the sorted combination of indices and dataset labels
        for index, dataset_label in sorted_combined:
            if dataset_label == "train":
                # Calculate the original index in the train dataset
                original_index = train_indices.index(index)
                concatenated_dataset.append(train_dataset[original_index])
            else:
                # Calculate the original index in the test dataset
                original_index = test_indices.index(index)
                concatenated_dataset.append(test_dataset[original_index])

        return TensorDataset(
            torch.tensor([[x[0]] for x in concatenated_dataset]),
            torch.tensor([[x[1]] for x in concatenated_dataset]),
        )
