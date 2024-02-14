import sys


if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self