from agogos.transformer import Transformer
from epochalyst._core._logging._logger import _Logger


class TransformationBlock(Transformer, _Logger):
    """The transformation block is a flexible block that allows for transformation of any data.
    
    To use this block, you must inherit from it and implement the following methods:
    - `transform`
    - `log_to_terminal`
    - `log_to_debug`
    - `log_to_warning`
    - `log_to_external`
    - `external_define_metric`
    """
