import numpy as np
import soundfile as sf
import torch

from ct.dit.model import ConditionalFlowMatching, DITModelConfig


class Inference:
    def __init__(self, config: DITModelConfig, model: ConditionalFlowMatching):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.config = config

    def process_batch(self, gen_text, ref_text, audio, speed, rms):
        pass
