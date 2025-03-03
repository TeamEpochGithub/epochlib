"""The base module contains the Base class."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from joblib import hash


@dataclass
class Base:
    """The Base class is the base class for all classes in the epochlib package.

    Methods:
    .. code-block:: python
        def get_hash(self) -> str:
            # Get the hash of base

        def get_parent(self) -> Any:
            # Get the parent of base.

        def get_children(self) -> list[Any]:
            # Get the children of base

        def save_to_html(self, file_path: Path) -> None:
            # Save html format to file_path
    """

    def __post_init__(self) -> None:
        """Initialize the block."""
        self.set_hash("")
        self.set_parent(None)
        self.set_children([])

    def set_hash(self, prev_hash: str) -> None:
        """Set the hash of the block.

        :param prev_hash: The hash of the previous block.
        """
        self._hash = hash(prev_hash + str(self))

    def get_hash(self) -> str:
        """Get the hash of the block.

        :return: The hash of the block.
        """
        return self._hash

    def get_parent(self) -> Any:
        """Get the parent of the block.

        :return: Parent of the block
        """
        return self._parent

    def get_children(self) -> Sequence[Any]:
        """Get the children of the block.

        :return: Children of the block
        """
        return self._children

    def save_to_html(self, file_path: Path) -> None:
        """Write html representation of class to file.

        :param file_path: File path to write to
        """
        html = self._repr_html_()
        with open(file_path, "w") as file:
            file.write(html)

    def set_parent(self, parent: Any) -> None:
        """Set the parent of the block.

        :param parent: Parent of the block
        """
        self._parent = parent

    def set_children(self, children: Sequence[Any]) -> None:
        """Set the children of the block.

        :param children: Children of the block
        """
        self._children = children

    def _repr_html_(self) -> str:
        """Return representation of class in html format.

        :return: String representation of html
        """
        html = "<div style='border: 1px solid black; padding: 10px;'>"
        html += f"<p><strong>Class:</strong> {self.__class__.__name__}</p>"
        html += "<ul>"
        html += f"<li><strong>Hash:</strong> {self.get_hash()}</li>"
        html += f"<li><strong>Parent:</strong> {self.get_parent()}</li>"
        html += "<li><strong>Children:</strong> "
        if self.get_children():
            html += "<ul>"
            for child in self.get_children():
                html += f"<li>{child._repr_html_()}</li>"
            html += "</ul>"
        else:
            html += "None"
        html += "</li>"
        html += "</ul>"
        html += "</div>"
        return html
