"""Verbatim CUA chat template: serve-time rendering that matches training.

Training renders conversations with :func:`omegalax.data.qwen3_encoding.build_chatml_text`:
raw ChatML, assistant content **verbatim** (which may contain literal
``<think>...</think>\\n{action}`` text), and a generation that continues from
``<|im_start|>assistant\\n``.

The stock Qwen3-VL-Thinking chat template does neither: it force-opens a think
block in the generation prompt (``<|im_start|>assistant\\n<think>\\n``) and
strips ``<think>...</think>`` from prior assistant turns
(``content.split('</think>')[-1]``). Serving an SFT checkpoint with the stock
template therefore feeds the model a token stream it was never trained on --
this train/serve mismatch silently broke evals for weeks.

``chat_template_verbatim.json`` (packaged next to this module) is the stock
template with exactly three changes:

1. assistant message content is rendered verbatim -- no ``</think>``
   splitting/stripping for any turn (and consequently no
   ``reasoning_content`` / last-query-index bookkeeping);
2. the ``add_generation_prompt`` tail is ``<|im_start|>assistant\\n`` with no
   ``<think>\\n``;
3. nothing else -- system handling, multimodal placeholder rendering
   (``<|vision_start|><|image_pad|><|vision_end|>``), and tool branches are
   byte-identical to stock.

``tests/test_chat_template_verbatim.py`` pins the byte-identity between
``apply_chat_template`` under this template and ``build_chatml_text``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

VERBATIM_CHAT_TEMPLATE_PATH = Path(__file__).with_name("chat_template_verbatim.json")

# Serving-side assets shipped with a HF checkpoint. Copied verbatim from the
# base-model snapshot; config.json is intentionally absent (the exporter writes
# its own) and so are the weight files.
TOKENIZER_ASSET_FILES = (
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "chat_template.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "generation_config.json",
)


def load_verbatim_chat_template() -> str:
    """Return the verbatim CUA chat template as a Jinja source string."""
    return json.loads(VERBATIM_CHAT_TEMPLATE_PATH.read_text())["chat_template"]


def copy_tokenizer_assets(base_model_dir: str | Path, out_dir: str | Path) -> list[Path]:
    """Copy tokenizer/processor assets from a base-model snapshot into ``out_dir``.

    Makes an exported checkpoint directory self-contained for sglang /
    transformers serving so nobody has to hand-copy files from the base
    snapshot afterwards (which would silently reintroduce the stock chat
    template). Returns the destination paths that were written.
    """
    base = Path(base_model_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for name in TOKENIZER_ASSET_FILES:
        src = base / name
        if src.exists():
            dst = out / name
            shutil.copyfile(src, dst)
            copied.append(dst)
    return copied


def write_chat_template(out_dir: str | Path, template: str) -> list[Path]:
    """Install ``template`` as the chat template of the checkpoint in ``out_dir``.

    Writes ``chat_template.json`` (the file processors read) and, if a
    ``tokenizer_config.json`` is present, replaces its ``chat_template`` key
    with the same string so tokenizer-only loaders agree with the processor.
    Returns the paths that were written.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    chat_template_path = out / "chat_template.json"
    chat_template_path.write_text(json.dumps({"chat_template": template}, indent=2) + "\n")
    written.append(chat_template_path)

    tokenizer_config_path = out / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        tokenizer_config = json.loads(tokenizer_config_path.read_text())
        tokenizer_config["chat_template"] = template
        tokenizer_config_path.write_text(json.dumps(tokenizer_config, indent=2) + "\n")
        written.append(tokenizer_config_path)

    return written
