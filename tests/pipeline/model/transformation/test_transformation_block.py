import pytest
from epochalyst.pipeline.model.transformation.transformation_block import TransformationBlock


class TestTransformationBlock:

    def test_transformation_block_init(self):
        tb = TransformationBlock()
        assert tb is not None

    def test_transformation_block_transform(self):
        with pytest.raises(NotImplementedError):
            tb = TransformationBlock()
            tb.transform(1)

    def test_transformation_block_log_to_terminal(self):
        with pytest.raises(NotImplementedError):
            tb = TransformationBlock()
            tb.log_to_terminal("test")

    def test_transformation_block_log_to_debug(self):
        with pytest.raises(NotImplementedError):
            tb = TransformationBlock()
            tb.log_to_debug("test")

    def test_transformation_block_log_to_warning(self):
        with pytest.raises(NotImplementedError):
            tb = TransformationBlock()
            tb.log_to_warning("test")
    
    def test_transformation_block_log_to_external(self):
        with pytest.raises(NotImplementedError):
            tb = TransformationBlock()
            tb.log_to_external("test")

    def test_transformation_block_external_define_metric(self):
        with pytest.raises(NotImplementedError):
            tb = TransformationBlock()
            tb.external_define_metric("test", "test")

