from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class StreamingCache:
    def __init__(self):
        self.cache = {}

    def get(self, layer_id: str, sample_indices: torch.Tensor):
        states = []
        max_length = 0

        # First pass:collect states and find max length
        for idx in sample_indices.tolist():
            key = (layer_id, idx)
            if key not in self.cache:
                return None
            state = self.cache[key]
            states.append(state)
            max_length = max(max_length, state.shape[-1])

        # Second pass: pad states to max length if needed
        if len(states) > 0 and states[0].dim() >= 2:
            padded_states = []
            for state in states:
                if state.shape[-1] < max_length:
                    pad_size = max_length - state.shape[-1]
                    padded_state = F.pad(state, (pad_size, 0), mode="constant", value=0)
                    padded_states.append(padded_state)

                else:
                    padded_states.append(state)
            return torch.stack(padded_states, dim=0)

        return torch.stack(states, dim=0)

    def set(self, layer_id: str, sample_indices: torch.Tensor, states: torch.Tensor):
        for i, idx in enumerate(sample_indices.tolist()):
            key = (layer_id, idx)
            self.cache[key] = states[i].detach()

    def set_to_zero(self, sample_indices: torch.Tensor):
        for key in list(self.cache.keys()):
            layer_id, sample_idx = key
            if sample_idx in sample_indices.tolist():
                cached_tensor = self.cache[key]
                self.cache[key] = torch.zeros_like(cached_tensor)

    def clear(
        self,
        layer_id: Optional[str] = None,
        sample_indices: Optional[torch.Tensor] = None,
    ):
        if layer_id is None and sample_indices is None:
            self.cache.clear()
        elif layer_id is not None and sample_indices is None:
            keys_to_remove = [k for k in self.cache.keys() if k[0] == layer_id]
            for k in keys_to_remove:
                del self.cache[k]
        elif layer_id is not None and sample_indices is not None:
            for idx in sample_indices.tolist():
                key = (layer_id, idx)
                self.cache.pop(key, None)
