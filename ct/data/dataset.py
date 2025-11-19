import io
from typing import Dict, List

import torch
import torchaudio
from datasets import Audio, load_dataset
from torch.utils.data import DataLoader, Dataset

TARGET_SR = 24000


class AudioDataset(Dataset):
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx) -> Dict:
        item = self.hf_dataset[idx]
        return {
            "text": item["text"],
            "audio_bytes": item["audio"]["bytes"],
        }


def load_and_resample_audio(audio_bytes_list: List[bytes]) -> torch.Tensor:
    waveforms = []
    for byte_data in audio_bytes_list:
        waveform, sr = torchaudio.load(io.BytesIO(byte_data))

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != TARGET_SR:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
            waveform = resampler(waveform)

        waveforms.append(waveform)

    lengths = [w.shape[1] for w in waveforms]
    max_len = max(lengths)

    padded = torch.zeros(len(waveforms), 1, max_len, dtype=waveforms[0].dtype)
    for i, wave in enumerate(waveforms):
        padded[i, :, : lengths[i]] = wave

    return padded


def collate_fn(batch: List[Dict]) -> Dict:

    texts = [item["text"] for item in batch]
    audio_bytes_list = [item["audio_bytes"] for item in batch]

    audio_tensor = load_and_resample_audio(audio_bytes_list)

    return {
        "text": texts,
        "audio": audio_tensor,
    }


def get_loader(batch_size):
    data = load_dataset("hf-internal-testing/librispeech_asr_demo", split="validation")
    data = data.cast_column("audio", Audio(decode=False))
    dataset = AudioDataset(data)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    return loader
