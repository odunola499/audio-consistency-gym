from ct.dit.model import ConditionalFlowMatching
from ct.dit.config import DITModelConfig
from ct.tokenizer.text.char_tokenizer import CharTokenizer
import torch
from ct.trainers.cfm_trainer import TrainModule, train_model
from ct.data.dataset import get_loader

print('imported loaders')

config = DITModelConfig()
tokenizer = CharTokenizer()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loader = get_loader(batch_size = 4)
train_module = TrainModule(
    config, loader
)

train_model(train_module)

