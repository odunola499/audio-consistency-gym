from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple


@dataclass
class DITModelConfig:
    num_layers: int = 2
    num_heads: int = 2
    head_dim: int = 8
    embed_dim: int = 16
    dropout: int = 0.1
    ff_mult: int = 4
    text_dim: int = 16
    conv_layers: int = 2
    vocab_size: int = 100
    vae_dim: int = 64
    checkpoint_activations: bool = False
    sigma: int = 0.0
    audio_drop_prob: float = 0.3
    cond_drop_prob: float = 0.2
    frac_lengths_mask: Tuple[float] = (0.1, 0.7)
    vae_hf_url: str = "odunola/vibevoice_vae_weights"
    max_duration: int = 4096

    lr: float = 1e-3
    epochs: Optional[int] = None
    learning_rate: float = 1e-4
    max_steps: Optional[int] = 20000
    warmup_steps: int = 2000
    keep_last_n_checkpoints: int = 2
    ckpt_dir: str = "checkpoints"
    pretrained_ckpt: Optional[str] = None
    resume_run: bool = True
    batch_size: int = 1
    grad_accumulation_steps: Optional[int] = 4
    max_grad_norm: int = 1.0
    noise_scheduler: Optional[str] = None
    log_to: Literal["wandb", "csv"] = "comet"
    wandb_project: str = "F5_TTS"
    wandb_run_name: Optional[str] = None
    log_samples: bool = True
    optimizer: Literal["bnb", "adamw"] = "bnb"
    lr_scheduler: Literal["cosine_warmup", "linear_lr", "sequential_lr"] = (
        "cosine_warmup"
    )
    save_interval: Optional[int] = 1000
    val_interval: Optional[int] = 1000
