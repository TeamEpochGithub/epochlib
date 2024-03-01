from agogos.trainer import Trainer
from epochalyst._core._logging._logger import _Logger


class TrainingBlock(Trainer, _Logger):
    """The training block is a flexible block that allows for training of any model.
    
    To use this block, you must inherit from it and implement the following methods:
    - `train`
    - `predict`
    - `log_to_terminal`
    - `log_to_debug`
    - `log_to_warning`
    - `log_to_external`
    - `external_define_metric`
    """

    