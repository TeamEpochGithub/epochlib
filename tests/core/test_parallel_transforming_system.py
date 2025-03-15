import pytest

from epochlib.core import Trainer, Transformer, TransformingSystem, ParallelTransformingSystem


class TestParallelTransformingSystem:
    def test_parallel_transforming_system(self):
        # Create an instance of the system
        system = ParallelTransformingSystem()

        # Assert the system is an instance of ParallelTransformingSystem
        assert isinstance(system, ParallelTransformingSystem)
        assert system is not None

    def test_parallel_transforming_system_wrong_step(self):
        class SubTransformer:
            def transform(self, x):
                return x

        with pytest.raises(TypeError):
            ParallelTransformingSystem(steps=[SubTransformer()])

    def test_parallel_transforming_system_transformers(self):
        transformer1 = Transformer()
        transformer2 = TransformingSystem()

        system = ParallelTransformingSystem(steps=[transformer1, transformer2])
        assert system is not None

    def test_parallel_transforming_system_transform(self):
        class transformer(Transformer):
            def transform(self, data):
                return data

        class pts(ParallelTransformingSystem):
            def concat(self, data1, data2, weight):
                if data1 is None:
                    return data2
                return data1 + data2

        t1 = transformer()

        system = pts(steps=[t1])

        assert system is not None
        assert system.transform([1, 2, 3]) == [1, 2, 3]

    def test_pts_transformers_transform(self):
        class transformer(Transformer):
            def transform(self, data):
                return data

        class pts(ParallelTransformingSystem):
            def concat(self, data1, data2, weight):
                if data1 is None:
                    return data2
                return data1 + data2

        t1 = transformer()
        t2 = transformer()

        system = pts(steps=[t1, t2])

        assert system is not None
        assert system.transform([1, 2, 3]) == [1, 2, 3, 1, 2, 3]

    def test_parallel_transforming_system_concat_throws_error(self):
        system = ParallelTransformingSystem()

        with pytest.raises(NotImplementedError):
            system.concat([1, 2, 3], [4, 5, 6])

    def test_pts_step_1_changed(self):
        system = ParallelTransformingSystem()

        t1 = Trainer()
        system.steps = [t1]

        with pytest.raises(TypeError):
            system.transform([1, 2, 3])

    def test_pts_step_2_changed(self):
        class pts(ParallelTransformingSystem):
            def concat(self, data1, data2, weight):
                if data1 is None:
                    return data2
                return data1 + data2

        system = pts()

        class transformer(Transformer):
            def transform(self, data):
                return data

        t1 = transformer()
        t2 = Trainer()
        system.steps = [t1, t2]

        with pytest.raises(TypeError):
            system.transform([1, 2, 3])

    def test_transform_parallel_hashes(self):
        class SubTransformer1(Transformer):
            def transform(self, x):
                return x

        class SubTransformer2(Transformer):
            def transform(self, x):
                return x * 2

        block1 = SubTransformer1()
        block2 = SubTransformer2()

        system1 = ParallelTransformingSystem(steps=[block1, block2])
        system1_copy = ParallelTransformingSystem(steps=[block1, block2])
        system2 = ParallelTransformingSystem(steps=[block2, block1])
        system2_copy = ParallelTransformingSystem(steps=[block2, block1])

        assert system1.get_hash() == system2.get_hash()
        assert system1.get_hash() == system1_copy.get_hash()
        assert system2.get_hash() == system2_copy.get_hash()
