"""No-follow and concurrent-mutation tests for artifact hashing."""

import os
import tempfile
from pathlib import Path
from unittest import mock

from absl.testing import absltest

from omegalax.data.artifact_contract import file_identity, validate_sha256


class ArtifactFileIdentityTest(absltest.TestCase):
    def test_uppercase_digest_rejects(self):
        with self.assertRaisesRegex(ValueError, "exact SHA-256"):
            validate_sha256("A" * 64, "digest")

    def test_symlink_rejects(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target = root / "target"
            target.write_bytes(b"payload")
            link = root / "link"
            link.symlink_to(target)

            with self.assertRaisesRegex(ValueError, "symlink"):
                file_identity(link)

    def test_mutation_during_hash_rejects(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "artifact"
            path.write_bytes(b"a" * (2 * 1024 * 1024))
            real_read = os.read
            mutated = False

            def mutate_after_first_read(fd: int, count: int) -> bytes:
                nonlocal mutated
                chunk = real_read(fd, count)
                if chunk and not mutated:
                    mutated = True
                    with path.open("r+b") as f:
                        f.seek(-1, os.SEEK_END)
                        f.write(b"b")
                return chunk

            with (
                mock.patch(
                    "omegalax.data.artifact_contract.os.read",
                    side_effect=mutate_after_first_read,
                ),
                self.assertRaisesRegex(ValueError, "changed while hashing"),
            ):
                file_identity(path)


if __name__ == "__main__":
    absltest.main()
