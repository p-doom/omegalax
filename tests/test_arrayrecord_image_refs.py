"""Tests for ArrayRecord-backed image references in Qwen encoding."""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from absl.testing import absltest
from array_record.python.array_record_module import ArrayRecordWriter
from PIL import Image

from omegalax.data import qwen3_encoding
from omegalax.data.qwen3_encoding import extract_images


def _jpeg_bytes(color: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 6), color).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


class ArrayRecordImageRefsTest(absltest.TestCase):
    def tearDown(self):
        qwen3_encoding._close_arrayrecord_image_sources()
        super().tearDown()

    def test_extract_images_reads_arrayrecord_uri(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shard_path = Path(tmpdir) / "images.array_record"
            writer = ArrayRecordWriter(str(shard_path), "group_size:1")
            try:
                writer.write(_jpeg_bytes((255, 0, 0)))
                writer.write(_jpeg_bytes((0, 255, 0)))
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
            self.assertEqual(images[0].mode, "RGB")
            self.assertEqual(images[0].size, (8, 6))

    def test_arrayrecord_reader_cache_evicts_and_closes_old_reader(self):
        old_cache_size = qwen3_encoding._ARRAYRECORD_IMAGE_CACHE_SIZE
        qwen3_encoding._ARRAYRECORD_IMAGE_CACHE_SIZE = 1
        qwen3_encoding._close_arrayrecord_image_sources()
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
                reader_a = qwen3_encoding._ARRAYRECORD_IMAGE_SOURCES[str(shard_a)]

                extract_images(
                    [
                        {
                            "role": "user",
                            "content": [{"type": "image", "image": f"ar://{shard_b.as_posix()}#0"}],
                        }
                    ]
                )

                self.assertFalse(reader_a.is_open())
                self.assertNotIn(str(shard_a), qwen3_encoding._ARRAYRECORD_IMAGE_SOURCES)
                self.assertIn(str(shard_b), qwen3_encoding._ARRAYRECORD_IMAGE_SOURCES)
        finally:
            qwen3_encoding._ARRAYRECORD_IMAGE_CACHE_SIZE = old_cache_size
            qwen3_encoding._close_arrayrecord_image_sources()


if __name__ == "__main__":
    absltest.main()
