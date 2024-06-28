from epochalyst.training.utils.recursive_repr import recursive_repr


class TestRecursiveRepr:
    def test_recursive_repr_with_dict(self):
        class TestClass:
            def __init__(self):
                self.attr1 = "value1"
                self.attr2 = "value2"

        obj = TestClass()
        expected_repr = "TestClass(attr1=value1, attr2=value2)"
        assert recursive_repr(obj) == expected_repr

    def test_recursive_repr_with_nested_objects(self):
        class NestedClass:
            def __init__(self):
                self.attr3 = "value3"

        class TestClass:
            def __init__(self):
                self.attr1 = "value1"
                self.attr2 = NestedClass()

        obj = TestClass()
        expected_repr = "TestClass(attr1=value1, attr2=NestedClass(attr3=value3))"
        assert recursive_repr(obj) == expected_repr

    def test_recursive_repr_with_truncated_value(self):
        class TestClass:
            def __init__(self):
                self.sound_file_paths = ["/data/raw/year/audio/file.wav"]

        obj = TestClass()
        expected_repr = "TestClass(sound_file_paths=data/raw/year/audio)"
        assert recursive_repr(obj) == expected_repr

    def test_recursive_repr_with_no_dict(self):
        obj = 123
        expected_repr = repr(obj)
        assert recursive_repr(obj) == expected_repr
