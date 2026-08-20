"""Tests for ArrayRecord-backed image references in Qwen encoding."""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
from absl.testing import absltest
from array_record.python.array_record_module import ArrayRecordWriter
from PIL import Image

from omegalax.data import arrayrecord_images
from omegalax.data.arrayrecord_images import extract_images


def _jpeg_bytes(color: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 6), color).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


class ArrayRecordImageRefsTest(absltest.TestCase):
    def tearDown(self):
        arrayrecord_images._close_arrayrecord_image_sources()
        super().tearDown()

    def test_extract_images_reads_the_arrayrecord_uri_fragment_as_the_record_index(self):
        record_0 = _jpeg_bytes((255, 0, 0))
        record_1 = _jpeg_bytes((0, 255, 0))
        with tempfile.TemporaryDirectory() as tmpdir:
            shard_path = Path(tmpdir) / "images.array_record"
            writer = ArrayRecordWriter(str(shard_path), "group_size:1")
            try:
                writer.write(record_0)
                writer.write(record_1)
            finally:
                writer.close()

            messages = [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": f"ar://{shard_path.as_posix()}#1"}],
                }
            ]
            images = extract_images(messages)

            self.assertLen(images, 1)
            # Which record came back, not just that some 8x6 RGB image did: two
            # records of identical geometry, so a reader that ignores the ``#1``
            # selector and always returns record 0 passes every shape assertion.
            expected = np.asarray(Image.open(io.BytesIO(record_1)).convert("RGB"))
            np.testing.assert_array_equal(np.asarray(images[0]), expected)
            self.assertFalse(
                np.array_equal(
                    expected, np.asarray(Image.open(io.BytesIO(record_0)).convert("RGB"))
                ),
                "the two records must be distinguishable for this test to bite",
            )

    def test_arrayrecord_reader_cache_evicts_and_closes_old_reader(self):
        old_cache_size = arrayrecord_images._ARRAYRECORD_IMAGE_CACHE_SIZE
        arrayrecord_images._ARRAYRECORD_IMAGE_CACHE_SIZE = 1
        arrayrecord_images._close_arrayrecord_image_sources()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                shard_a = Path(tmpdir) / "a.array_record"
                shard_b = Path(tmpdir) / "b.array_record"
                for shard_path, color in [(shard_a, (255, 0, 0)), (shard_b, (0, 255, 0))]:
                    writer = ArrayRecordWriter(str(shard_path), "group_size:1")
                    try:
                        writer.write(_jpeg_bytes(color))
                    finally:
                        writer.close()

                extract_images(
                    [
                        {
                            "role": "user",
                            "content": [{"type": "image", "image": f"ar://{shard_a.as_posix()}#0"}],
                        }
                    ]
                )
                reader_a = arrayrecord_images._ARRAYRECORD_IMAGE_SOURCES[str(shard_a)]

                extract_images(
                    [
                        {
                            "role": "user",
                            "content": [{"type": "image", "image": f"ar://{shard_b.as_posix()}#0"}],
                        }
                    ]
                )

                self.assertFalse(reader_a.is_open())
                self.assertNotIn(str(shard_a), arrayrecord_images._ARRAYRECORD_IMAGE_SOURCES)
                self.assertIn(str(shard_b), arrayrecord_images._ARRAYRECORD_IMAGE_SOURCES)
        finally:
            arrayrecord_images._ARRAYRECORD_IMAGE_CACHE_SIZE = old_cache_size
            arrayrecord_images._close_arrayrecord_image_sources()


if __name__ == "__main__":
    absltest.main()
