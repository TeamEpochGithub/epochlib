
import glob
import os


def remove_cache_files():
    files = glob.glob("tests/cache/*")
    for f in files:
        # If f is readme.md, skip it
        if "readme.md" in f:
            continue
        os.remove(f)