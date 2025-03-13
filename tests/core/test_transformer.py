import pytest

from epochlib.core import Transformer


class TestTransformer:
    def test_transformer_abstract(self):
        transformer = Transformer()

        with pytest.raises(NotImplementedError):
            transformer.transform([1, 2, 3])

    def test_transformer_transform(self):
        class transformerInstance(Transformer):
            def transform(self, data):
                return data

        transformer = transformerInstance()

        assert transformer.transform([1, 2, 3]) == [1, 2, 3]

    def test_transformer_hash(self):
        transformer = Transformer()
        assert transformer.get_hash() == "1cbcc4f2d0921b050d9b719d2beb6529"
