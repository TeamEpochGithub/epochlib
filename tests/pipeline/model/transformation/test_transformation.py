from epochalyst.pipeline.model.transformation.transformation import (
    TransformationPipeline,
)


class TestTransformationPipeline:
    def test_transformation_pipeline_init(self):
        tp = TransformationPipeline()
        assert tp.steps is not None

    def test_transformation_pipeline_transform(self):
        tp = TransformationPipeline()
        x = 1
        assert tp.transform(x) == x
        assert tp.transform(x, transform_args={"a": 1}) == x
        assert tp.transform(x, transform_args={"a": 1, "b": 2}) == x

    def test_transformation_pipeline_with_steps(self):
        class TestTransformationBlock(TransformationPipeline):
            def transform(self, x, **transform_args):
                return x

        t1 = TestTransformationBlock()
        t2 = TestTransformationBlock()
        tp = TransformationPipeline(steps=[t1, t2])

        assert tp.transform(None) is None
