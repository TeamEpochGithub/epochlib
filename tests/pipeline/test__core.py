from epochlib.pipeline import _Block, _Base, _SequentialSystem, _ParallelSystem
from tests.util import remove_cache_files
from pathlib import Path


class Test_Base:
    def test_init(self):
        base = _Base()
        assert base is not None

    def test_get_hash(self):
        assert _Base().get_hash() == "be5a33685928d3da88062f187295a019"

    def test_set_hash(self):
        base = _Base()
        prev_hash = base.get_hash()
        base._set_hash("prev_hash")
        assert base.get_hash() != prev_hash

    def test_get_children(self):
        base = _Base()
        assert base.get_children() == []

    def test_get_parent(self):
        base = _Base()
        assert base.get_parent() is None

    def test__set_parent(self):
        base = _Base()
        base._set_parent(base)
        assert base.get_parent() == base

    def test__set_children(self):
        base = _Base()
        base._set_children([base])
        assert base.get_children() == [base]

    def test__repr_html_(self):
        base = _Base()
        assert (
            base._repr_html_()
            == "<div style='border: 1px solid black; padding: 10px;'><p><strong>Class:</strong> _Base</p><ul><li><strong>Hash:</strong> be5a33685928d3da88062f187295a019</li><li><strong>Parent:</strong> None</li><li><strong>Children:</strong> None</li></ul></div>"
        )

    def test_save_to_html(self):
        html_path = Path("./tests/cache/test_html.html")
        Path("./tests/cache/").mkdir(parents=True, exist_ok=True)
        base = _Base()
        base.save_to_html(html_path)
        assert Path.exists(html_path)
        remove_cache_files()


class TestBlock:
    def test_block_init(self):
        block = _Block()
        assert block is not None

    def test_block_set_hash(self):
        block = _Block()
        block._set_hash("")
        hash1 = block.get_hash()
        assert hash1 == "04714d9ee40c9baff8c528ed982a103c"
        block._set_hash(hash1)
        hash2 = block.get_hash()
        assert hash2 == "83196595c42f8eff9218c0ac8f80faf0"
        assert hash1 != hash2

    def test_block_get_hash(self):
        block = _Block()
        block._set_hash("")
        hash1 = block.get_hash()
        assert hash1 == "04714d9ee40c9baff8c528ed982a103c"

    def test__repr_html_(self):
        block_instance = _Block()

        html_representation = block_instance._repr_html_()

        assert html_representation is not None


class TestSequentialSystem:
    def test_system_init(self):
        system = _SequentialSystem()
        assert system is not None

    def test_system_hash_no_steps(self):
        system = _SequentialSystem()
        assert system.get_hash() == ""

    def test_system_hash_with_1_step(self):
        block1 = _Block()

        system = _SequentialSystem([block1])
        assert system.get_hash() == "04714d9ee40c9baff8c528ed982a103c"
        assert block1.get_hash() == system.get_hash()

    def test_system_hash_with_2_steps(self):
        block1 = _Block()
        block2 = _Block()

        system = _SequentialSystem([block1, block2])
        assert system.get_hash() != block1.get_hash()
        assert (
            system.get_hash() == block2.get_hash() == "83196595c42f8eff9218c0ac8f80faf0"
        )

    def test_system_hash_with_3_steps(self):
        block1 = _Block()
        block2 = _Block()
        block3 = _Block()

        system = _SequentialSystem([block1, block2, block3])
        assert system.get_hash() != block1.get_hash()
        assert system.get_hash() != block2.get_hash()
        assert block1.get_hash() != block2.get_hash()
        assert (
            system.get_hash() == block3.get_hash() == "5aaa5f0962baedf36f132ad39380761e"
        )

    def test__repr_html_(self):
        block_instance = _Block()
        system_instance = _SequentialSystem([block_instance, block_instance])
        html_representation = system_instance._repr_html_()

        assert html_representation is not None


class TestParallelSystem:
    def test_parallel_system_init(self):
        parallel_system = _ParallelSystem()
        assert parallel_system is not None

    def test_parallel_system_hash_no_steps(self):
        system = _ParallelSystem()
        assert system.get_hash() == ""

    def test_parallel_system_hash_with_1_step(self):
        block1 = _Block()

        system = _ParallelSystem([block1])
        assert system.get_hash() == "04714d9ee40c9baff8c528ed982a103c"
        assert block1.get_hash() == system.get_hash()

    def test_parallel_system_hash_with_2_steps(self):
        block1 = _Block()
        block2 = _Block()

        system = _ParallelSystem([block1, block2])
        assert system.get_hash() != block1.get_hash()
        assert block1.get_hash() == block2.get_hash()
        assert system.get_hash() != block2.get_hash()
        assert system.get_hash() == "9689e0f292013df811f8e910684406f7"

    def test_parallel_system_hash_with_3_steps(self):
        block1 = _Block()
        block2 = _Block()
        block3 = _Block()

        system = _ParallelSystem([block1, block2, block3])
        assert system.get_hash() != block1.get_hash()
        assert system.get_hash() != block2.get_hash()
        assert system.get_hash() != block3.get_hash()
        assert block1.get_hash() == block2.get_hash() == block3.get_hash()
        assert system.get_hash() == "b5ea75f99dbfb82c35e082c88b94bda7"

    def test_parallel_system__repr_html_(self):
        block_instance = _Block()
        system_instance = _ParallelSystem([block_instance, block_instance])
        html_representation = system_instance._repr_html_()

        assert html_representation is not None
