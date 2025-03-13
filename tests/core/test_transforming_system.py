import warnings
import numpy as np
import pytest

from epochlib.core import Transformer, TransformingSystem


class TestTransformingSystem:
    def test_transforming_system_init(self):
        transforming_system = TransformingSystem()
        assert transforming_system is not None

    def test_transforming_system_init_with_steps(self):
        class SubTransformer(Transformer):
            def transform(self, x):
                return x

        block1 = SubTransformer()
        transforming_system = TransformingSystem(steps=[block1])
        assert transforming_system is not None

    def test_transforming_system_wrong_step(self):
        class SubTransformer:
            def transform(self, x):
                return x

        with pytest.raises(TypeError):
            TransformingSystem(steps=[SubTransformer()])

    def test_transforming_system_steps_changed(self):
        class SubTransformer:
            def transform(self, x):
                return x

        block1 = SubTransformer()
        transforming_system = TransformingSystem()
        transforming_system.steps = [block1]
        with pytest.raises(TypeError):
            transforming_system.transform([1, 2, 3])

    def test_transforming_system_transform_1_block(self):
        class SubTransformer(Transformer):
            def transform(self, x):
                return x

        block1 = SubTransformer()
        transforming_system = TransformingSystem(steps=[block1])
        assert transforming_system.transform([1, 2, 3]) == [1, 2, 3]

    def test_transforming_system_transform_1_block_with_args(self):
        class SubTransformer(Transformer):
            def transform(self, data):
                return data

        block1 = SubTransformer()
        transforming_system = TransformingSystem(steps=[block1])
        assert transforming_system.transform([1, 2, 3], **{"SubTransformer": {}}) == [
            1,
            2,
            3,
        ]

    def test_transforming_system_transform_2_blocks(self):
        class SubTransformer(Transformer):
            def transform(self, x):
                return x * 2

        block1 = SubTransformer()
        block2 = SubTransformer()
        transforming_system = TransformingSystem(steps=[block1, block2])
        result = transforming_system.transform(np.array([1, 2, 3]))
        assert np.array_equal(result, np.array([4, 8, 12]))

    def test_transformsys_with_transformsys(self):
        class SubTransformer(Transformer):
            def transform(self, x):
                return x * 2

        block1 = SubTransformer()
        block2 = TransformingSystem(steps=[block1])
        transforming_system = TransformingSystem(steps=[block2])
        result = transforming_system.transform(np.array([1, 2, 3]))
        assert np.array_equal(result, np.array([2, 4, 6]))

    def test_transforming_system_transform_with_args(self):
        class SubTransformer(Transformer):
            def transform(self, data, multiplier=2):
                return data * multiplier

        block1 = SubTransformer()
        transforming_system = TransformingSystem(steps=[block1])
        result = transforming_system.transform(
            np.array([1, 2, 3]), **{"SubTransformer": {"multiplier": 2}}
        )
        assert np.array_equal(result, np.array([2, 4, 6]))

    def test_transforming_system_transform_with_args_2_blocks(self):
        class SubTransformer(Transformer):
            def transform(self, data, multiplier=2):
                return data * multiplier

        block1 = SubTransformer()
        block2 = SubTransformer()
        transforming_system = TransformingSystem(steps=[block1, block2])
        result = transforming_system.transform(
            np.array([1, 2, 3]), **{"SubTransformer": {"multiplier": 2}}
        )
        assert np.array_equal(result, np.array([4, 8, 12]))

    def test_transforming_system_transform_with_recursive_args(self):
        class SubTransformer(Transformer):
            def transform(self, data, multiplier=2):
                return data * multiplier

        block1 = SubTransformer()
        block2 = SubTransformer()
        block3 = TransformingSystem(steps=[block2])
        block4 = TransformingSystem(steps=[block3])
        transforming_system = TransformingSystem(steps=[block1, block4])
        assert np.array_equal(
            transforming_system.transform(
                np.array([1, 2, 3]), **{"SubTransformer": {"multiplier": 2}}
            ),
            np.array([4, 8, 12]),
        )
        assert np.array_equal(
            transforming_system.transform(
                np.array([1, 2, 3]),
                **{
                    "TransformingSystem": {
                        "TransformingSystem": {"SubTransformer": {"multiplier": 3}}
                    }
                },
            ),
            np.array([6, 12, 18]),
        )

    def test_transforming_system_empty_hash(self):
        transforming_system = TransformingSystem()
        assert transforming_system.get_hash() == ""

    def test_transforming_system_wrong_kwargs(self):
        class Block1(Transformer):
            def transform(self, x, **kwargs):
                return x

        class Block2(Transformer):
            def transform(self, x, **kwargs):
                return x

        block1 = Block1()
        block2 = Block2()
        system = TransformingSystem(steps=[block1, block2])
        kwargs = {"Block1": {}, "block2": {}}
        with pytest.warns(
            UserWarning,
            match="The following steps do not exist but were given in the kwargs:",
        ):
            system.transform([1, 2, 3], **kwargs)

    def test_transforming_system_right_kwargs(self):
        class Block1(Transformer):
            def transform(self, x, **kwargs):
                return x

        class Block2(Transformer):
            def transform(self, x, **kwargs):
                return x

        block1 = Block1()
        block2 = Block2()
        system = TransformingSystem(steps=[block1, block2])
        kwargs = {"Block1": {}, "Block2": {}}
        with warnings.catch_warnings(record=True) as caught_warnings:
            system.transform([1, 2, 3], **kwargs)

        assert not caught_warnings
