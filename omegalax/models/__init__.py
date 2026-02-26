"""omegalax.models — Model implementations for Qwen3, Qwen3-VL, and Qwen3.5.

Dimension Key (Shazeer shape-suffix notation)
=============================================

Tensor variable names end with a suffix of uppercase letters denoting their
shape dimensions.  For example, ``hidden_BTD`` is a 3-D tensor with batch,
sequence-length, and model-dimension axes.

Text
---------------
B — batch size
T — sequence length (target / current tokens)
S — source / memory sequence length (KV-cache or attended-to length)
D — model / hidden dimension (hidden_size, emb_dim)
V — vocabulary size
F — feed-forward intermediate size (mlp_dim, intermediate_size, moe_intermediate_size)
H — number of attention heads
K — head dimension (head_dim, size of each attention key or value)
G — number of KV-head groups (num_kv_heads, for GQA)
R — GQA repetition factor (num_heads // num_kv_heads)
E — number of MoE experts

Vision
-------------------------
C — input channels
N — number of vision tokens (after patch embedding / merging)
W — vision output hidden dimension (when different from D)
P — patch spatial dimension

DeltaNet (module-local, documented in deltanet.py)
---------------------------------------------------------------------
J — number of chunks
L — chunk length
A — linear-attention key head dim
U — linear-attention value head dim
Uses B, H, K from the core key where applicable.

MRoPE
-----
Z — the 3-dimensional MRoPE axis (temporal, height, width = 3)
"""

from .qwen3 import registry

__all__ = ["registry"]
