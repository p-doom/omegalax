"""Qwen chat-template encoding for text and vision-language SFT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from transformers import BaseImageProcessor, PreTrainedTokenizer

from omegalax.data.arrayrecord_images import extract_images


@dataclass(frozen=True)
class _TurnEncoding:
    input_ids: np.ndarray
    loss_mask: np.ndarray
    final_input_ids: np.ndarray | None = None
    final_loss_mask: np.ndarray | None = None


@dataclass(frozen=True)
class _EncodedConversation:
    input_ids: np.ndarray
    loss_mask: np.ndarray
    pixel_values: np.ndarray | None
    image_grid_thw: np.ndarray
    measurements: list[dict[str, Any]]


def _token_ids(tokenizer: PreTrainedTokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False, return_attention_mask=False)
    return list(encoded["input_ids"])


def _render(tokenizer: PreTrainedTokenizer, messages: list[dict[str, Any]]) -> str:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    if not isinstance(rendered, str):
        raise TypeError("tokenizer.apply_chat_template() must return text")
    return rendered


def _validate_messages(messages: list[dict[str, Any]], *, multimodal: bool) -> None:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    if not isinstance(messages[0], dict):
        raise TypeError("messages[0] must be an object")
    if not multimodal:
        for message in messages:
            if not isinstance(message, dict) or not isinstance(message.get("content"), list):
                continue
            for part in message["content"]:
                if isinstance(part, dict) and (
                    part.get("type") in {"image", "image_url"}
                    or "image" in part
                    or "image_url" in part
                ):
                    raise ValueError("image content requires an image processor")

    expected_role = "system" if messages[0].get("role") == "system" else "user"
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or set(message) - {"role", "content", "reasoning_content"}:
            raise ValueError(f"messages[{index}] has unsupported fields")
        role = message.get("role")
        if role != expected_role:
            raise ValueError(f"messages[{index}].role must be {expected_role!r}, got {role!r}")
        if role == "system":
            expected_role = "user"
        elif role == "user":
            expected_role = "assistant"
        else:
            expected_role = "user"

        reasoning_content = message.get("reasoning_content")
        if reasoning_content is not None and (
            role != "assistant" or not isinstance(reasoning_content, str)
        ):
            raise ValueError("reasoning_content must be a string on an assistant turn")

        content = message.get("content")
        if isinstance(content, str):
            if "<|video_pad|>" in content:
                raise ValueError("video content is not supported")
            continue
        if not multimodal or not isinstance(content, list) or not content:
            expected = "a string" if not multimodal else "a string or non-empty part list"
            raise ValueError(f"messages[{index}].content must be {expected}")
        for part_index, part in enumerate(content):
            if not isinstance(part, dict):
                raise TypeError(f"messages[{index}].content[{part_index}] must be an object")
            if part.get("type") == "video" or "video" in part:
                raise ValueError("video content is not supported")
            part_type = part.get("type")
            if part_type == "text":
                if set(part) != {"type", "text"} or not isinstance(part["text"], str):
                    raise ValueError(
                        f"messages[{index}].content[{part_index}] must contain exactly "
                        "type and string text"
                    )
                if "<|video_pad|>" in part["text"]:
                    raise ValueError("video content is not supported")
            elif part_type in {"image", "image_url"}:
                if role != "user":
                    raise ValueError("image parts are supported only in user turns")
            else:
                raise ValueError(
                    f"messages[{index}].content[{part_index}] has unsupported type {part_type!r}"
                )


class QwenChatMessageEncoder:
    """Encode Qwen messages with the supplied tokenizer and image processor."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        image_processor: BaseImageProcessor | None,
    ) -> None:
        if not isinstance(tokenizer.chat_template, str) or not tokenizer.chat_template:
            raise ValueError("tokenizer must define a chat template")
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self._multimodal = image_processor is not None
        self._assistant_header = np.asarray(
            _token_ids(tokenizer, "<|im_start|>assistant\n"), dtype=np.int32
        )
        self._newline = np.asarray(_token_ids(tokenizer, "\n"), dtype=np.int32)
        self._im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self._image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self._video_token_id = tokenizer.convert_tokens_to_ids("<|video_pad|>")
        if not self._assistant_header.size or not self._newline.size:
            raise ValueError("tokenizer does not implement the Qwen ChatML boundary tokens")
        if not isinstance(self._im_end_id, int) or self._im_end_id < 0:
            raise ValueError("tokenizer does not define <|im_end|>")
        if self._multimodal and (
            not isinstance(self._image_token_id, int) or self._image_token_id < 0
        ):
            raise ValueError("multimodal tokenizer does not define <|image_pad|>")
        if self._multimodal and (
            not isinstance(self._video_token_id, int) or self._video_token_id < 0
        ):
            raise ValueError("multimodal tokenizer does not define <|video_pad|>")

    def _turn_texts(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[str], list[str | None], str]:
        dummy = {"role": "user", "content": ""}
        dummy_text = _render(self.tokenizer, [dummy])
        historical: list[str] = []
        final: list[str | None] = []

        for message in messages:
            role = message["role"]
            if role == "system":
                prefixed = _render(self.tokenizer, [message, dummy])
                if not prefixed.endswith(dummy_text):
                    raise ValueError("chat template does not preserve the system turn boundary")
                historical.append(prefixed[: -len(dummy_text)])
                final.append(None)
                continue
            if role == "user":
                historical.append(_render(self.tokenizer, [message]))
                final.append(None)
                continue

            middle = _render(self.tokenizer, [dummy, message, dummy])
            if not middle.startswith(dummy_text) or not middle.endswith(dummy_text):
                raise ValueError("chat template does not preserve Qwen turn boundaries")
            historical.append(middle[len(dummy_text) : -len(dummy_text)])

            terminal = _render(self.tokenizer, [dummy, message])
            if not terminal.startswith(dummy_text):
                raise ValueError("chat template does not preserve the final assistant boundary")
            final.append(terminal[len(dummy_text) :])

        full_text = _render(self.tokenizer, messages)
        selected = [
            final_text if index == len(messages) - 1 and final_text is not None else turn_text
            for index, (turn_text, final_text) in enumerate(zip(historical, final, strict=True))
        ]
        if "".join(selected) != full_text:
            raise ValueError("chat template cannot be decomposed into Qwen message turns")
        return historical, final, full_text

    def _assistant_mask(self, ids: np.ndarray) -> np.ndarray:
        header_len = len(self._assistant_header)
        footer_len = 1 + len(self._newline)
        if len(ids) < header_len + footer_len:
            raise ValueError("assistant turn is shorter than its ChatML boundaries")
        if not np.array_equal(ids[:header_len], self._assistant_header):
            raise ValueError("assistant turn does not start with the Qwen ChatML header")
        if ids[-footer_len] != self._im_end_id or not np.array_equal(
            ids[-len(self._newline) :], self._newline
        ):
            raise ValueError("assistant turn does not end with the Qwen ChatML stop token")
        mask = np.zeros(len(ids), dtype=np.int32)
        mask[header_len : len(ids) - len(self._newline)] = 1
        return mask

    def _expand_images(
        self,
        ids: np.ndarray,
        grids: np.ndarray,
    ) -> np.ndarray:
        image_positions = np.flatnonzero(ids == self._image_token_id)
        if len(image_positions) != len(grids):
            raise ValueError(
                "chat template image placeholders do not match the structured image parts"
            )
        if not len(grids):
            return ids

        merge_size = int(self.image_processor.merge_size)
        pieces: list[np.ndarray] = []
        offset = 0
        for position, grid in zip(image_positions, grids, strict=True):
            pieces.append(ids[offset:position])
            token_count = int(np.prod(grid, dtype=np.int64)) // (merge_size * merge_size)
            if token_count <= 0:
                raise ValueError("image processor produced an empty image grid")
            pieces.append(np.full(token_count, self._image_token_id, dtype=np.int32))
            offset = int(position) + 1
        pieces.append(ids[offset:])
        return np.concatenate(pieces)

    def _encode_turn(
        self,
        text: str,
        role: str,
        grids: np.ndarray,
        final_text: str | None,
    ) -> _TurnEncoding:
        ids = self._expand_images(
            np.asarray(_token_ids(self.tokenizer, text), dtype=np.int32), grids
        )
        mask = self._assistant_mask(ids) if role == "assistant" else np.zeros(len(ids), np.int32)
        if final_text is None:
            return _TurnEncoding(ids, mask)

        final_ids = self._expand_images(
            np.asarray(_token_ids(self.tokenizer, final_text), dtype=np.int32), grids
        )
        return _TurnEncoding(
            ids,
            mask,
            final_ids,
            self._assistant_mask(final_ids),
        )

    def _encode(self, messages: list[dict[str, Any]]) -> _EncodedConversation:
        _validate_messages(messages, multimodal=self._multimodal)
        images_by_turn = [extract_images([message]) for message in messages]
        images = [image for turn_images in images_by_turn for image in turn_images]
        if images and self.image_processor is None:
            raise ValueError("image content requires an image processor")

        pixel_values = None
        if images:
            processed = self.image_processor(images=images, return_tensors="np")
            grids = np.asarray(processed["image_grid_thw"], dtype=np.int32).reshape(-1, 3)
            pixel_values = np.asarray(processed["pixel_values"])
            if len(grids) != len(images):
                raise ValueError("image processor output does not match the input image count")
        else:
            grids = np.empty((0, 3), dtype=np.int32)

        historical, terminal, full_text = self._turn_texts(messages)
        turn_encodings: list[_TurnEncoding] = []
        grid_offset = 0
        for message, text, final_text, turn_images in zip(
            messages, historical, terminal, images_by_turn, strict=True
        ):
            turn_grids = grids[grid_offset : grid_offset + len(turn_images)]
            grid_offset += len(turn_images)
            turn_encodings.append(self._encode_turn(text, message["role"], turn_grids, final_text))

        selected_ids = [
            turn.final_input_ids
            if index == len(turn_encodings) - 1 and turn.final_input_ids is not None
            else turn.input_ids
            for index, turn in enumerate(turn_encodings)
        ]
        input_ids = np.concatenate(selected_ids)
        full_ids = np.asarray(_token_ids(self.tokenizer, full_text), dtype=np.int32)
        if np.any(full_ids == self._video_token_id):
            raise ValueError("video content is not supported")
        expected_unexpanded_images = int(np.sum(full_ids == self._image_token_id))
        if expected_unexpanded_images != len(images):
            raise ValueError(
                "chat template image placeholders do not match the structured image parts"
            )
        expanded_full = self._expand_images(full_ids, grids)
        if not np.array_equal(input_ids, expanded_full):
            raise ValueError("turn encodings do not reproduce the full chat-template token stream")

        selected_masks = [
            turn.final_loss_mask
            if index == len(turn_encodings) - 1 and turn.final_loss_mask is not None
            else turn.loss_mask
            for index, turn in enumerate(turn_encodings)
        ]
        loss_mask = np.concatenate(selected_masks)
        measurements: list[dict[str, Any]] = []
        grid_offset = 0
        merge_size = int(getattr(self.image_processor, "merge_size", 1))
        for turn, turn_images in zip(turn_encodings, images_by_turn, strict=True):
            turn_grids = grids[grid_offset : grid_offset + len(turn_images)]
            grid_offset += len(turn_images)
            vision_patches = sum(int(np.prod(row, dtype=np.int64)) for row in turn_grids)
            vision_tokens = vision_patches // (merge_size * merge_size)
            final_ids = turn.final_input_ids
            final_mask = turn.final_loss_mask
            measurements.append(
                {
                    "length": len(turn.input_ids),
                    "terminal_length_delta": (
                        0 if final_ids is None else int(len(final_ids) - len(turn.input_ids))
                    ),
                    "supervised_tokens": int(turn.loss_mask.sum()),
                    "terminal_supervised_tokens_delta": (
                        0 if final_mask is None else int(final_mask.sum() - turn.loss_mask.sum())
                    ),
                    "vision_tokens": vision_tokens,
                    "vision_patches": vision_patches,
                    "num_images": len(turn_grids),
                    "image_grid_thw": turn_grids.tolist(),
                }
            )
        return _EncodedConversation(
            input_ids,
            loss_mask,
            pixel_values,
            grids,
            measurements,
        )

    def encode(self, messages: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        encoded = self._encode(messages)
        result = {
            "input_ids": encoded.input_ids,
            "loss_mask": encoded.loss_mask,
        }
        if encoded.pixel_values is not None:
            result["pixel_values"] = encoded.pixel_values
            result["image_grid_thw"] = encoded.image_grid_thw
        return result

    def measure(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self._encode(messages).measurements


class QwenConversationMeasurement:
    """Picklable conversation measurement callable for spawned workers."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        image_processor: BaseImageProcessor | None,
    ) -> None:
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.encoder = QwenChatMessageEncoder(tokenizer, image_processor)

    def reject_unmeasurable(self, messages: list[dict[str, Any]]) -> None:
        _validate_messages(messages, multimodal=self.image_processor is not None)

    def __call__(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.reject_unmeasurable(messages)
        return self.encoder.measure(messages)


def make_message_length_fn(
    tokenizer: PreTrainedTokenizer,
    image_processor: BaseImageProcessor | None = None,
) -> QwenConversationMeasurement:
    return QwenConversationMeasurement(tokenizer, image_processor)
