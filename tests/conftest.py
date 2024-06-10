import shutil

import pytest

from tests.constants import TEMP_DIR


@pytest.fixture
def setup_temp_dir():
    TEMP_DIR.mkdir(exist_ok=True)

    yield

    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
