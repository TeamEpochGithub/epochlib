from epochlib.core import Block, SequentialSystem


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