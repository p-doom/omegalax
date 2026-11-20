"""One-off: which files differ between the original tree and the working copy."""

import filecmp
import os

ORIGINAL = "/fast/home/salan.isaqzoi/gh_projects/p(doom)/omegalax/omegalax"
COPY = "/fast/home/salan.isaqzoi/gh_projects/p(doom)/omegalax_optimization/omegalax"


def walk(root):
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for name in filenames:
            if name.endswith(".pyc"):
                continue
            path = os.path.join(dirpath, name)
            out[os.path.relpath(path, root)] = path
    return out


left, right = walk(ORIGINAL), walk(COPY)
print("only in ORIGINAL:", sorted(set(left) - set(right)))
print("only in COPY    :", sorted(set(right) - set(left)))
print("differing:")
for key in sorted(set(left) & set(right)):
    if not filecmp.cmp(left[key], right[key], shallow=False):
        print("   ", key)
