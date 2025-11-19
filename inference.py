from ct.dit.model import ConditionalFlowMatching,  DITModelConfig
import torch
from safetensors.torch import load_file
from ct.tokenizer.text.char_tokenizer import CharTokenizer
from datasets import load_dataset, Audio
import soundfile as sf
import librosa
import io
import os

device = torch.device('cuda')
# data = load_dataset("hf-internal-testing/librispeech_asr_demo", split="validation")
# data = data.cast_column("audio", Audio(decode=False))
# sample = data[0]["audio"]
# audio, sr = librosa.load(io.BytesIO(sample["bytes"]), sr = 24000)


# sf.write("sample.wav", audio, sr)

audio_path = 'sample.wav'
audio, sr = librosa.load(audio_path, sr = 24000)
ref_text = "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL"
gen_text = "NOR IS MISTER QUILTER'S MANNER LESS INTERESTING THAN HIS MATTER"

config = DITModelConfig()
tokenizer = CharTokenizer()

model = ConditionalFlowMatching(config, tokenizer).to(device)

weights_path = '/root/audio-consistency-gym/checkpoints'
weights_path = [os.path.join(weights_path, i) for i in os.listdir(weights_path)][0]

state = load_file(weights_path)
state = {i.replace('ema_model.',''):j for i,j in state.items()}
model.load_state_dict(state, strict = False)

duration = 10
text = ref_text + gen_text
audio = torch.tensor(audio)[None, None, ...].to(device)

output, _ = model.sample(text = text, condition = audio, duration = duration)
print(output.shape)

sf.write('output.wav', output[0][0].detach().cpu().numpy(), 24000)

output, _ = model.acoustic_model(audio)
print(output.shape)

sf.write('new_output.wav', output[0][0].detach().cpu(), 24000)
