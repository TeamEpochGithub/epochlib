import os
import shutil


def remove_cache_files() -> None:
    """Remove all files and directories in the given path."""
    path = "tests/cache"
    for filename in os.listdir(path):
        if filename in ["README", "README.md"]:
            continue
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
