from epochlib.core import Block
from tests.core.util import remove_cache_files
from pathlib import Path


class TestBlock:
    def test_block_init(self):
        block = Block()
        assert block is not None

    def test_block_set_hash(self):
        block = Block()
        block.set_hash("")
        hash1 = block.get_hash()
        assert hash1 != ""
        block.set_hash(hash1)
        hash2 = block.get_hash()
        assert hash2 != ""
        assert hash1 != hash2

    def test_block_get_hash(self):
        block = Block()
        block.set_hash("")
        hash1 = block.get_hash()
        assert hash1 != ""

    def test__repr_html_(self):
        block_instance = Block()

        html_representation = block_instance._repr_html_()

        assert html_representation is not None