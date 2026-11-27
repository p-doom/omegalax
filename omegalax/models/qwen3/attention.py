import jax
from flax import nnx
from jax.sharding import PartitionSpec as P
from tokamax import dot_product_attention

from omegalax.models.kv_cache import LayerCache
from .norms import RMSNorm
from .rope import apply_rope, generate_pos_embeddings
from .utils import compute_positions_from_segment_ids

wp = nnx.with_partitioning


class Attention(nnx.Module):
    def __init__(self, cfg, *, rngs: nnx.Rngs):
        self.shd_cfg = cfg.shd_cfg
        self.dtype = cfg.dtype
        init_fn = nnx.initializers.lecun_normal()
        qkv_init = wp(init_fn, ("embed", "heads"))
        o_init = wp(init_fn, ("heads", "embed"))
        self.q_proj = nnx.Linear(
            cfg.emb_dim,
            cfg.num_heads * cfg.head_dim,
            use_bias=False,
            rngs=rngs,
            dtype=cfg.dtype,
            kernel_init=qkv_init,
        )
        self.k_proj = nnx.Linear(
            cfg.emb_dim,
            cfg.num_kv_heads * cfg.head_dim,
            use_bias=False,
            rngs=rngs,
            dtype=cfg.dtype,
            kernel_init=qkv_init,
        )
        self.v_proj = nnx.Linear(
            cfg.emb_dim,
            cfg.num_kv_heads * cfg.head_dim,
            use_bias=False,
            rngs=rngs,
            dtype=cfg.dtype,
            kernel_init=qkv_init,
        )
        self.o_proj = nnx.Linear(
            cfg.num_heads * cfg.head_dim,
            cfg.emb_dim,
            use_bias=False,
            rngs=rngs,
            dtype=cfg.dtype,
            kernel_init=o_init,
        )

        self.q_norm = RMSNorm(cfg.head_dim, cfg.norm_eps, rngs=rngs, sharding=(None,))
        self.k_norm = RMSNorm(cfg.head_dim, cfg.norm_eps, rngs=rngs, sharding=(None,))
        self.n_rep = cfg.num_heads // cfg.num_kv_heads
        self.scale = cfg.head_dim**-0.5
        self.head_dim = cfg.head_dim
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        object.__setattr__(self, "_q_sharding", None)
        object.__setattr__(self, "_q_sharding_spec", P(*cfg.shd_cfg.act_btnh))
        object.__setattr__(self, "_attn_backend", "mosaic_gpu")
        object.__setattr__(self, "_attn_kind", "text")

    @jax.named_scope("attention")
    def __call__(
        self, hidden_BTD: jax.Array, cache: LayerCache | None, segment_ids_BT: jax.Array
    ) -> tuple[jax.Array, LayerCache | None]:
        q_proj_BTF = self.q_proj(hidden_BTD, out_sharding=self.shd_cfg.act_btf)
        k_proj_BTF = self.k_proj(hidden_BTD, out_sharding=self.shd_cfg.act_btf)
        v_proj_BTF = self.v_proj(hidden_BTD, out_sharding=self.shd_cfg.act_btf)
        B, T = hidden_BTD.shape[:2]
        q_BTHK = self.q_norm(
            jax.lax.reshape(
                q_proj_BTF,
                (B, T, self.num_heads, self.head_dim),
                out_sharding=self.shd_cfg.act_btnh,
            )
        )
        k_BTGK = self.k_norm(
            jax.lax.reshape(
                k_proj_BTF,
                (B, T, self.num_kv_heads, self.head_dim),
                out_sharding=self.shd_cfg.act_btnh,
            )
        )
        v_BTGK = jax.lax.reshape(
            v_proj_BTF, (B, T, self.num_kv_heads, self.head_dim), out_sharding=self.shd_cfg.act_btnh
        )

        if cache is None:
            positions_BT = compute_positions_from_segment_ids(segment_ids_BT)
        else:
            positions_BT = cache.batch.positions_BT
        sin_BTK, cos_BTK = generate_pos_embeddings(positions_BT, self.head_dim)
        sin_BTK = sin_BTK.astype(self.dtype)
        cos_BTK = cos_BTK.astype(self.dtype)
        q_BTHK = apply_rope(q_BTHK, sin_BTK, cos_BTK)
        k_BTGK = apply_rope(k_BTGK, sin_BTK, cos_BTK)

        B, T, H, K = q_BTHK.shape
        if cache is None:
            attn_BTHK = dot_product_attention(
                q_BTHK,
                k_BTGK,
                v_BTGK,
                is_causal=True,
                scale=self.scale,
                implementation=self._attn_backend,
                q_sharding=self._q_sharding,
            )
        else:
            cache = cache.write(k_BTGK, v_BTGK)
            attn_BTHK = cache.attend(q_BTHK, self.scale)
        out_BTD = self.o_proj(
            jax.lax.reshape(attn_BTHK, (B, T, self.num_heads * K), out_sharding=self.shd_cfg.act_btf),
            out_sharding=self.shd_cfg.act_btd,
        )
        return out_BTD, cache
