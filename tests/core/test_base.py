from epochlib.core import Base
from tests.core.util import remove_cache_files
from pathlib import Path

class Test_Base:
    def test_init(self):
        base = Base()
        assert base is not None

    def test_set_hash(self):
        base = Base()
        prev_hash = base.get_hash()
        base.set_hash("prev_hash")
        assert base.get_hash() != prev_hash

    def test_get_children(self):
        base = Base()
        assert base.get_children() == []

    def test_get_parent(self):
        base = Base()
        assert base.get_parent() is None

    def test__set_parent(self):
        base = Base()
        base.set_parent(base)
        assert base.get_parent() == base

    def test__set_children(self):
        base = Base()
        base.set_children([base])
        assert base.get_children() == [base]

    def test__repr_html_(self):
        base = Base()
        assert (
            base._repr_html_()
            == "<div style='border: 1px solid black; padding: 10px;'><p><strong>Class:</strong> Base</p><ul><li><strong>Hash:</strong> a00a595206d7eefcf0e87acf6e2e22ee</li><li><strong>Parent:</strong> None</li><li><strong>Children:</strong> None</li></ul></div>"
        )

    def test_save_to_html(self):
        html_path = Path("./tests/cache/test_html.html")
        Path("./tests/cache/").mkdir(parents=True, exist_ok=True)
        base = Base()
        base.save_to_html(html_path)
        assert Path.exists(html_path)
        remove_cache_files()