import random
from typing import Optional, Union, List
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ct.dit.config import DITModelConfig
from ct.dit.modules import DiT
from ct.dit.block import SemanticConnector
from ct.dit.utils import (
    load_vae_models,
    lens_to_mask,
    mask_from_frac_lengths,
    get_epss_timesteps,
    manual_euler,
)
from ct.tokenizer.text.char_tokenizer import CharTokenizer


class ConditionalFlowMatching(nn.Module):
    def __init__(self, config: DITModelConfig, tokenizer: CharTokenizer):
        super().__init__()

        self.transformer = DiT(
            dim=config.embed_dim,
            vocab_size=tokenizer.vocab_size,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            dropout=config.dropout,
            ff_mult=config.ff_mult,
            vae_dim=config.vae_dim,
            text_dim=config.text_dim,
            conv_layers=config.conv_layers,
        )
        self.embed_dim = config.embed_dim
        self.sigma = config.sigma
        self.frac_lengths_mask = config.frac_lengths_mask
        self.audio_drop_prob = config.audio_drop_prob
        self.cond_drop_prob = config.cond_drop_prob
        self.tokenizer = CharTokenizer()

        self.acoustic_model, self.semantic_model = load_vae_models(repo_id = config.vae_hf_url)
        self.acoustic_model.requires_grad_(False)
        self.semantic_model.requires_grad_(False)

        self.semantic_connector = SemanticConnector(
            self.semantic_model.config.vae_dim, config.vae_dim
        )
        self.config = config

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def get_latents(self, audio: Tensor):
        _, semantics = self.semantic_model(audio, debug=False)
        encoder_output = self.acoustic_model(audio, debug=False)
        acoustics, _ = self.acoustic_model.sampling(encoder_output)
        return semantics, acoustics

    def latent_to_audio(self, acoustic_latents: torch.Tensor):
        return self.acoustic_model.decode(acoustic_latents)

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        text: Union[torch.Tensor, List[str]],
        duration,
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        use_epss=True,
    ):
        semantics, acoustics = self.get_latents(condition)
        condition = semantics + acoustics
        batch, cond_seq_len = condition.shape[:2]
        device = condition.device

        lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)
        if isinstance(text, list):
            text, attn_mask = self.tokenizer(text)
            text = text.to(device, dtype=torch.long)
        cond_mask = lens_to_mask(lens)

        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)

        duration = torch.maximum(
            torch.maximum((text != -1).sum(dim=-1), lens) + 1, duration
        )
        duration = duration.clamp(max=self.config.max_duration)
        max_duration = duration.amax()
        condition = F.pad(condition, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
        cond_mask = F.pad(
            cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False
        )
        cond_mask = cond_mask.unsqueeze(-1)
        step_cond = torch.where(cond_mask, condition, torch.zeros_like(condition))

        if batch > 1:
            mask = lens_to_mask(duration)
        else:
            mask = None

        def fn(t, x):
            if cfg_strength < 1e-5:
                pred = self.transformer(
                    x=x,
                    cond=step_cond,
                    text=text,
                    time=t,
                    mask=mask,
                    drop_audio_cond=False,
                    drop_text=False,
                    cache=True,
                )
                return pred

            pred_cfg = self.transformer(
                x=x,
                cond=step_cond,
                text=text,
                time=t,
                mask=mask,
                cfg_infer=True,
                cache=True,
            )
            pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
            return pred + (pred - null_pred) * cfg_strength

        y0 = []
        for dur in duration:
            y0.append(
                torch.randn(
                    dur, self.num_channels, device=self.device, dtype=step_cond.dtype
                )
            )
        y0 = torch.nn.utils.rnn.pad_sequence(y0, padding_value=0, batch_first=True)

        t_start = 0

        if t_start == 0 and use_epss:
            t = get_epss_timesteps(steps, device=self.device, dtype=step_cond.dtype)
        else:
            t = torch.linspace(
                t_start, 1, steps + 1, device=self.device, dtype=step_cond.dtype
            )
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        trajectory = manual_euler(fn, y0, t)
        self.transformer.clear_cache()

        sampled = trajectory[-1]
        out = sampled
        out = torch.where(cond_mask, condition, out)
        out = self.latent_to_audio(out)
        return out, trajectory

    def forward(
        self,
        text: Union[torch.Tensor, List[str]],
        audio: Optional[Tensor] = None,
        acoustic_latents: Optional[Tensor] = None,
        semantic_latents: Optional[Tensor] = None,
    ):
        if audio is None:
            assert (
                acoustic_latents is not None
            ), "acoustic latents should be present if no audio"
            assert (
                semantic_latents is not None
            ), "semantic latents should be present if no audio"

        else:
            semantic_latents, acoustic_latents = self.get_latents(audio)

        semantic_latents = self.semantic_connector(semantic_latents)
        device = semantic_latents.device
        dtype = semantic_latents.dtype

        batch_size, seq_len = acoustic_latents.shape[:2]

        if isinstance(text, list):
            text, attn_mask = self.tokenizer(text)
            text = text.to(device, dtype=torch.long)
            attn_mask = attn_mask.to(device, dtype=torch.long)

        lens = torch.full((batch_size,), seq_len, device=device)
        mask = lens_to_mask(lens, length=seq_len)

        frac_lengths = torch.empty(
            batch_size, device=device, dtype=torch.float
        ).uniform_(*self.frac_lengths_mask)
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)

        rand_span_mask &= mask

        inp = semantic_latents + acoustic_latents
        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(inp), inp)
        acoustic_segment = torch.where(
            rand_span_mask[..., None], acoustic_latents, torch.zeros_like(inp)
        )
        x1 = cond + acoustic_segment

        x0 = torch.randn_like(x1)
        t = torch.rand((batch_size,), dtype=dtype, device=device)[..., None, None]
        x_t = (1 - t) * x0 + t * x1
        flow = x1 - x0

        drop_audio_cond = random.random() < self.audio_drop_prob
        if random.random() < self.cond_drop_prob:
            drop_audio_cond = True
            drop_text = True
        else:
            drop_text = False

        pred = self.transformer(
            x=x_t,
            cond=cond,
            text=text,
            time=t,
            drop_audio_cond=drop_audio_cond,
            drop_text=drop_text,
            mask=mask,
        )

        loss = F.mse_loss(pred, flow, reduction="none")
        loss = loss[rand_span_mask]
        return loss.mean(), cond, pred


if __name__ == "__main__":
    batch_size = 4
    seq_len = 15
    frac_lengths_mask = (0.1, 0.7)
    embed_dim = 16
    x1 = torch.randn((batch_size, seq_len, embed_dim))

    lens = torch.full((batch_size,), seq_len)
    mask = lens_to_mask(lens, seq_len)
    frac_lengths = torch.zeros((batch_size,)).float().uniform_(*frac_lengths_mask)
    rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)

    rand_span_mask &= mask

    print(rand_span_mask[0])
    inp = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)
    print(inp[0])
