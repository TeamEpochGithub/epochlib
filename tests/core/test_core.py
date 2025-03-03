from epochlib.core import Block, Base, SequentialSystem, ParallelSystem
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


class TestSequentialSystem:
    def test_system_init(self):
        system = SequentialSystem()
        assert system is not None

    def test_system_hash_no_steps(self):
        system = SequentialSystem()
        assert system.get_hash() == ""

    def test_system_hash_with_1_step(self):
        block1 = Block()

        system = SequentialSystem([block1])
        assert system.get_hash() != ""
        assert block1.get_hash() == system.get_hash()

    def test_system_hash_with_2_steps(self):
        block1 = Block()
        block2 = Block()

        system = SequentialSystem([block1, block2])
        assert system.get_hash() != block1.get_hash()
        assert (
            system.get_hash() == block2.get_hash() != ""
        )

    def test_system_hash_with_3_steps(self):
        block1 = Block()
        block2 = Block()
        block3 = Block()

        system = SequentialSystem([block1, block2, block3])
        assert system.get_hash() != block1.get_hash()
        assert system.get_hash() != block2.get_hash()
        assert block1.get_hash() != block2.get_hash()
        assert (
            system.get_hash() == block3.get_hash() != ""
        )

    def test__repr_html_(self):
        block_instance = Block()
        system_instance = SequentialSystem([block_instance, block_instance])
        html_representation = system_instance._repr_html_()

        assert html_representation is not None


class TestParallelSystem:
    def test_parallel_system_init(self):
        parallel_system = ParallelSystem()
        assert parallel_system is not None

    def test_parallel_system_hash_no_steps(self):
        system = ParallelSystem()
        assert system.get_hash() == ""

    def test_parallel_system_hash_with_1_step(self):
        block1 = Block()

        system = ParallelSystem([block1])
        assert system.get_hash() != ""
        assert block1.get_hash() == system.get_hash()

    def test_parallel_system_hash_with_2_steps(self):
        block1 = Block()
        block2 = Block()

        system = ParallelSystem([block1, block2])
        assert system.get_hash() != block1.get_hash()
        assert block1.get_hash() == block2.get_hash()
        assert system.get_hash() != block2.get_hash()
        assert system.get_hash() != ""

    def test_parallel_system_hash_with_3_steps(self):
        block1 = Block()
        block2 = Block()
        block3 = Block()

        system = ParallelSystem([block1, block2, block3])
        assert system.get_hash() != block1.get_hash()
        assert system.get_hash() != block2.get_hash()
        assert system.get_hash() != block3.get_hash()
        assert block1.get_hash() == block2.get_hash() == block3.get_hash()
        assert system.get_hash() != ""

    def test_parallel_system__repr_html_(self):
        block_instance = Block()
        system_instance = ParallelSystem([block_instance, block_instance])
        html_representation = system_instance._repr_html_()

        assert html_representation is not None
