import torch
from torch import Tensor, nn
from torch.nn import functional as F


def precompute_freqs(dim, end, theta=10000, theta_rescale_factor=1.0):
    theta *= theta_rescale_factor ** (dim / (dim - 2))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return torch.cat([freqs_cos, freqs_sin], dim=-1)


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNextV2Block(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int, dilation: int = 1):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: Tensor):
        residual = x
        x = x.transpose(1, 2)
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x + residual
        return x


class TextEmbedding(nn.Module):
    def __init__(self, vocab_size, dim, mask_padding=True, conv_layers=4, conv_mult=2):
        super().__init__()
        self.precompute_max_pos = 4096

        self.text_embed = nn.Embedding(vocab_size + 1, dim)
        self.pos_embed = nn.Embedding(self.precompute_max_pos, dim)
        self.mask_padding = mask_padding

        self.register_buffer(
            "freq_cis", precompute_freqs(dim, self.precompute_max_pos), persistent=False
        )
        self.text_blocks = nn.Sequential(
            *[ConvNextV2Block(dim, dim * conv_mult) for _ in range(conv_layers)]
        )

    def upsample_text_by_mask(self, text, text_mask, audio_mask):
        batch_size, text_len, text_dim = text.shape

        if audio_mask is None:
            audio_mask = torch.ones_like(text_mask, dtype=torch.bool)

        valid_mask = audio_mask & text_mask
        audio_lens = audio_mask.sum(dim=1)
        valid_lens = valid_mask.sum(dim=1)

        upsampled_text = torch.zeros_like(text)

        for i in range(batch_size):
            audio_len = audio_lens[i].item()
            valid_len = valid_lens[i].item()
            if valid_len == 0:
                continue

            valid_ind = torch.where(valid_mask[i])[0]
            valid_data = text[i, valid_ind, :]

            base_repeat = audio_len // valid_len
            remainder = audio_len % valid_len

            indices = []
            for j in range(valid_len):
                repeat_count = base_repeat + (1 if j >= valid_len - remainder else 0)
                indices.extend([j] * repeat_count)

            indices = torch.tensor(
                indices[:audio_len], device=text.device, dtype=torch.long
            )
            upsampled = valid_data[indices]

            upsampled_text[i, :audio_len, :] = upsampled
        return upsampled_text

    def forward(self, text: Tensor, seq_len, drop_text=False, audio_mask=None):
        text_mask = None
        text = text + 1
        text = text[:, :seq_len]
        batch, text_len = text.shape[0], text.shape[1]
        text = F.pad(text, (0, seq_len - text_len), value=0)
        if self.mask_padding:
            text_mask = text == 0

        if drop_text:  # Classifier free Guidance
            text = torch.zeros_like(text)

        text = self.text_embed(text)
        pos = torch.arange(0, seq_len, device=text.device, dtype=torch.long)
        pos_embed = self.pos_embed(pos)
        text += pos_embed

        if self.mask_padding:
            text = text.masked_fill(
                text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0
            )
            for block in self.text_blocks:
                text = block(text)
                text = text.masked_fill(
                    text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0
                )
        else:
            text = self.text_blocks(text)

        return text


if __name__ == "__main__":
    from tokenizer.text.char_tokenizer import CharTokenizer

    tokenizer = CharTokenizer()
    texts = ["my name is odunola", "I am a boy."]
    input_ids, attn_mask = tokenizer(texts)

    embed = TextEmbedding(vocab_size=tokenizer.vocab_size, dim=8)
    output = embed(input_ids, seq_len=input_ids.shape[-1])
    print(input_ids.shape)
    print(output.shape)
    print(tokenizer.vocab_size)
