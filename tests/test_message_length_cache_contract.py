"""Fail-closed identity tests for the message-length cache."""

import json
import tempfile
from pathlib import Path

from absl.testing import absltest

from omegalax.data.grain_pipeline import (
    _load_chat_message_lengths,
    _validate_chat_message_lengths,
    _write_chat_message_lengths,
)


def _contract(tokenizer_digest: str = "a" * 64) -> dict:
    return {
        "version": 1,
        "tokenizer_sha256": tokenizer_digest,
        "processor_sha256": None,
        "preprocessor_sha256": None,
    }


def _measurement(length: int = 1) -> dict:
    return {
        "length": length,
        "terminal_length_delta": 0,
        "supervised_tokens": 0,
        "terminal_supervised_tokens_delta": 0,
        "vision_tokens": 0,
        "vision_patches": 0,
        "num_images": 0,
        "image_grid_thw": [],
    }


class MessageLengthCacheContractTest(absltest.TestCase):
    def _write_chat(self, path: Path, contents: list[str]) -> None:
        with path.open("w") as f:
            for content in contents:
                f.write(
                    json.dumps(
                        {
                            "messages": [
                                {"role": "user", "content": content},
                                {"role": "assistant", "content": "ok"},
                            ]
                        }
                    )
                    + "\n"
                )

    def test_identical_chat_and_measurement_contract_accept(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chat = Path(tmpdir) / "chat.jsonl"
            cache = Path(tmpdir) / "message_lengths.jsonl"
            self._write_chat(chat, ["aa", "bb"])
            measurements = {
                (row, offset): _measurement() for row in range(2) for offset in range(2)
            }

            _write_chat_message_lengths(cache, measurements, chat, _contract())
            header, loaded = _load_chat_message_lengths(cache)

            _validate_chat_message_lengths(chat, header, loaded, _contract())
            self.assertEqual(loaded, measurements)

    def test_same_shape_changed_bytes_reject(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chat = Path(tmpdir) / "chat.jsonl"
            cache = Path(tmpdir) / "message_lengths.jsonl"
            self._write_chat(chat, ["aa", "bb"])
            measurements = {
                (row, offset): _measurement() for row in range(2) for offset in range(2)
            }
            _write_chat_message_lengths(cache, measurements, chat, _contract())
            header, loaded = _load_chat_message_lengths(cache)

            self._write_chat(chat, ["az", "bb"])

            with self.assertRaisesRegex(ValueError, "source chat identity"):
                _validate_chat_message_lengths(chat, header, loaded, _contract())

    def test_reordered_rows_reject(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chat = Path(tmpdir) / "chat.jsonl"
            cache = Path(tmpdir) / "message_lengths.jsonl"
            self._write_chat(chat, ["aa", "bb"])
            measurements = {
                (row, offset): _measurement() for row in range(2) for offset in range(2)
            }
            _write_chat_message_lengths(cache, measurements, chat, _contract())
            header, loaded = _load_chat_message_lengths(cache)

            self._write_chat(chat, ["bb", "aa"])

            with self.assertRaisesRegex(ValueError, "source chat identity"):
                _validate_chat_message_lengths(chat, header, loaded, _contract())

    def test_changed_tokenizer_contract_reject(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chat = Path(tmpdir) / "chat.jsonl"
            cache = Path(tmpdir) / "message_lengths.jsonl"
            self._write_chat(chat, ["aa"])
            measurements = {(0, 0): _measurement(), (0, 1): _measurement()}
            _write_chat_message_lengths(cache, measurements, chat, _contract())
            header, loaded = _load_chat_message_lengths(cache)

            with self.assertRaisesRegex(ValueError, "measurement contract"):
                _validate_chat_message_lengths(
                    chat,
                    header,
                    loaded,
                    _contract(tokenizer_digest="e" * 64),
                )

    def test_legacy_headerless_cache_rejects(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Path(tmpdir) / "message_lengths.jsonl"
            cache.write_text(json.dumps({"conv_idx": 0, "msg_offset": 0, "measurement": 1}) + "\n")

            with self.assertRaisesRegex(TypeError, "versioned header"):
                _load_chat_message_lengths(cache)


if __name__ == "__main__":
    absltest.main()
