import jax
import jax.numpy as jnp
import yaml
import dataclasses
from pathlib import Path
from typing import Callable, Optional, Tuple
from jax.sharding import Mesh
from utils import jax_pytree_struct


AxisName = str | tuple[str, ...] | None
Axes = tuple[AxisName, ...]

# Expected physical mesh axis names:
# x - batch
# y - 1st of 2D tensor sharding
# z - 2nd of 2D tensor sharding
BATCH_AXIS_NAME = "x"
EXPERT_AXIS_NAME = "z"
TENSOR_ONLY_AXIS_NAME = "y"
ATTN_HEADS_AXIS_NAME = "y"
TENSOR_AXIS_NAME = ("y", "z")


def init_uniform(scale=1.0):
    def kernel_init(key, shape, dtype):
        return jax.random.uniform(key, shape, dtype, minval=-scale, maxval=scale)

    return kernel_init


@dataclasses.dataclass
class EmbeddingConfig:
    dtype: jnp.dtype = jnp.bfloat16
    vocab_size: int = 50304
    d_emb: int = 768
    num_layers: int = 12
    weight_initializer: Callable = dataclasses.field(init=False)
    weight_logical_axes: Tuple[str, str] = ("embed_in", "embed_out")

    def __post_init__(self):
        self.weight_initializer = jax.nn.initializers.normal(stddev=1.0)


@dataclasses.dataclass
class GroupedQueryAttentionConfig:
    dtype: jnp.dtype = jnp.bfloat16
    d_emb: int = 768
    q_heads: int = 8
    kv_heads: int = 4
    num_layers: int = 12
    head_dim: int = dataclasses.field(init=False)

    wq_initializer: Callable = dataclasses.field(init=False)
    wk_initializer: Callable = dataclasses.field(init=False)
    wv_initializer: Callable = dataclasses.field(init=False)
    wo_initializer: Callable = dataclasses.field(init=False)

    wq_logical_axes: Tuple[str, str, str] = ("attn_wqkv_in", "attn_q_heads", "attn_head_dim")
    wk_logical_axes: Tuple[str, str, str] = ("attn_wqkv_in", "attn_kv_heads", "attn_head_dim")
    wv_logical_axes: Tuple[str, str, str] = ("attn_wqkv_in", "attn_kv_heads", "attn_head_dim")
    wo_logical_axes: Tuple[str, str, str] = ("attn_wo_in", "attn_head_dim", "attn_wo_out")

    def __post_init__(self):
        self.head_dim = self.d_emb // self.q_heads
        self.wq_initializer = init_uniform(scale=3**0.5 * self.d_emb**-0.5)
        self.wk_initializer = init_uniform(scale=3**0.5 * self.d_emb**-0.5)
        self.wv_initializer = init_uniform(scale=3**0.5 * self.d_emb**-0.5)
        self.wo_initializer = jax.nn.initializers.zeros


@dataclasses.dataclass
class LinearConfig:
    dtype: jnp.dtype = jnp.bfloat16
    in_features: int = 768
    out_features: int = 50304
    use_bias: bool = False
    weight_initializer: Callable = None
    weight_logical_axes: Tuple[str, str] = ("linear_in", "linear_out")


@dataclasses.dataclass
class MLPConfig:
    d_emb: int = 768
    dtype: jnp.dtype = jnp.bfloat16
    fc1: LinearConfig = dataclasses.field(init=False)
    fc2: LinearConfig = dataclasses.field(init=False)

    def __post_init__(self):
        self.fc1 = LinearConfig(
            dtype=self.dtype,
            in_features=self.d_emb,
            out_features=self.d_emb * 4,
            weight_initializer=init_uniform(scale=3**0.5 * self.d_emb**-0.5),
            weight_logical_axes=("mlp_fc1_in", "mlp_fc1_out"),
        )
        self.fc2 = LinearConfig(
            dtype=self.dtype,
            in_features=self.d_emb * 4,
            out_features=self.d_emb,
            weight_initializer=jax.nn.initializers.zeros,
            weight_logical_axes=("mlp_fc2_in", "mlp_fc2_out"),
        )


@dataclasses.dataclass
class ModelConfig:
    seqlen: int = 2048
    vocab_size: int = 50304
    d_emb: int = 768
    num_layers: int = 16
    q_heads: int = 8
    kv_heads: int = 4
    attn_impl: str = "flash_attn"
    dtype: jnp.dtype = jnp.bfloat16

    embed: EmbeddingConfig = dataclasses.field(init=False)
    mlp: MLPConfig = dataclasses.field(init=False)
    lm_head: LinearConfig = dataclasses.field(init=False)
    attn: GroupedQueryAttentionConfig = dataclasses.field(init=False)

    def __post_init__(self):
        self.embed = EmbeddingConfig(
            dtype=self.dtype,
            vocab_size=self.vocab_size,
            d_emb=self.d_emb,
            num_layers=self.num_layers,
        )
        self.attn = GroupedQueryAttentionConfig(
            dtype=self.dtype,
            d_emb=self.d_emb,
            q_heads=self.q_heads,
            kv_heads=self.kv_heads,
            num_layers=self.num_layers,
        )
        self.mlp = MLPConfig(dtype=self.dtype, d_emb=self.d_emb)
        self.lm_head = LinearConfig(
            dtype=self.dtype,
            in_features=self.d_emb,
            out_features=self.vocab_size,
            weight_initializer=jax.nn.initializers.normal(stddev=0.001),
        )


@dataclasses.dataclass
class ShardingRules:
    batch: AxisName = BATCH_AXIS_NAME
    sequence: AxisName = None
    act_embed: AxisName = None
    act_heads: AxisName = None

    embed_in: AxisName = None
    embed_out: AxisName = None

    attn_wqkv_in: AxisName = None
    attn_q_heads: AxisName = None
    attn_kv_heads: AxisName = None
    attn_head_dim: AxisName = None

    attn_wo_in: AxisName = None
    attn_wo_out: AxisName = None

    norm_in: AxisName = None
    norm_out: AxisName = None

    mlp_fc1_in: AxisName = None
    mlp_fc1_out: AxisName = None
    mlp_fc2_in: AxisName = None
    mlp_fc2_out: AxisName = None

    linear_in: AxisName = None
    linear_out: AxisName = None


@dataclasses.dataclass
class ProfileConfig:
    enabled: bool = False
    profile_dir: str = "profiles/"
    start_step: int = 5
    end_step: int = 10


@dataclasses.dataclass
class CheckpointConfig:
    # Checkpoint related
    max_checkpoints_to_keep: int = 5
    checkpoint_save_steps: int = 100
    last_checkpoint_step: int = 0
    # Directory where checkpoints will be saved
    save_ckpt_dir: Path | str = ""
    # Path to params subdirectory within a checkpoint from which weights will be loaded
    load_params_ckpt_path: Path | str = ""


@dataclasses.dataclass
class HyperParams:
    # Batch size related
    micro_batch_size: int = 32
    global_batch_size: int = 256
    grad_accum_steps: Optional[float] = dataclasses.field(init=False)

    # Optimizer related
    max_lr: float = 6e-4
    min_lr: float = 6e-5
    embedding_lr: float = 0.2
    unembedding_lr: float = 0.004
    b1: float = 0.8
    b2: float = 0.95
    weight_decay: float = 0.0
    cautious_weight_decay: float = 0.01
    clip_grad_norm: float = 1.0
    total_train_steps: int = 10000
    warmup_steps: int = int(min(300, 0.01 * total_train_steps))  # ~10% of total steps


DTYPE_MAP = {
    "float32": jnp.float32,
    "float16": jnp.float16,
    "bfloat16": jnp.bfloat16,
}


def _dtype_from_str(s: str) -> jnp.dtype:
    if s not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype '{s}'. Choose from {list(DTYPE_MAP)}")
    return DTYPE_MAP[s]


def _build_model_config(d: dict) -> "ModelConfig":
    d = dict(d)
    if "dtype" in d:
        d["dtype"] = _dtype_from_str(d["dtype"])
    return ModelConfig(**d)


def _build_hparams(d: dict) -> "HyperParams":
    d = dict(d)
    return HyperParams(**d)


def _build_checkpoint_config(d: dict) -> "CheckpointConfig":
    d = dict(d)
    return CheckpointConfig(**d)


def _build_profile_config(d: dict) -> "ProfileConfig":
    d = dict(d)
    return ProfileConfig(**d)


def load_config_from_yaml(
    path: str | Path,
    *,
    mesh: Mesh = None,
    rules: "ShardingRules" = None,
    seed: jax.Array = None,
) -> "Config":
    """Build a Config from a YAML file.

    The YAML file is expected to have top-level keys matching the Config
    sub-configs: ``model``, ``hparams``, ``checkpoint``, and optionally
    ``data_dir``.  Runtime objects (mesh, rules, seed) are passed as
    keyword arguments since they cannot be serialized.
    """
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    model_cfg = _build_model_config(raw.get("model", {}))
    hparams = _build_hparams(raw.get("hparams", {}))
    ckpt_cfg = _build_checkpoint_config(raw.get("checkpoint", {}))
    profile_cfg = _build_profile_config(raw.get("profile", {}))
    data_dir = raw.get("data_dir", "")

    if rules is None:
        rules = ShardingRules()

    return Config(
        seed=seed,
        mesh=mesh,
        rules=rules,
        model=model_cfg,
        hparams=hparams,
        ckpt_cfg=ckpt_cfg,
        profile_cfg=profile_cfg,
        data_dir=data_dir,
    )


@jax_pytree_struct
class Config:
    seed: jax.Array = None
    mesh: Mesh = None
    rules: ShardingRules = dataclasses.field(default_factory=ShardingRules)
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    hparams: HyperParams = dataclasses.field(default_factory=HyperParams)
    ckpt_cfg: CheckpointConfig = dataclasses.field(default_factory=CheckpointConfig)
    profile_cfg: ProfileConfig = dataclasses.field(default_factory=ProfileConfig)
    data_dir: Path | str = ""
