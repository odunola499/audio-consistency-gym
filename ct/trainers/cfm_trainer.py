import os
import random
from datetime import datetime

import lightning as pl
import torch
from ema_pytorch import EMA
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import CometLogger
from random_word import RandomWords
from safetensors.torch import save_file
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from ct.data.dataset import get_loader
from ct.dit.model import ConditionalFlowMatching, DITModelConfig
from ct.tokenizer.text.char_tokenizer import CharTokenizer

now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
word = RandomWords().get_random_word()

run_name = f"{now_str}_{word}"


class TrainModule(pl.LightningModule):
    def __init__(self, config: DITModelConfig, loader):
        super().__init__()
        self.model = None
        self.ema_model = None
        self.config = config
        self.train_loader = loader

    def configure_model(self) -> None:
        config = self.config
        tokenizer = CharTokenizer()
        self.model = ConditionalFlowMatching(config, tokenizer)
        self.ema_model = EMA(self.model, include_online_model=False)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.config.learning_rate)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.config.max_steps,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def save(self):
        directory = self.config.ckpt_dir
        save_n_files = self.config.keep_last_n_checkpoints
        os.makedirs(directory, exist_ok=True)

        state_dict = self.ema_model.state_dict()
        state_dict = {
            k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)
        }
        filename = os.path.join(directory, f"model_{self.global_step}.safetensors")

        save_file(state_dict, filename)
        ckpts = sorted(
            [
                os.path.join(directory, f)
                for f in os.listdir(directory)
                if f.startswith("model_") or f.startswith("adapter_")
            ],
            key=os.path.getmtime,
        )
        for ckpt in ckpts[:-save_n_files]:
            os.remove(ckpt)

        print(f"Saved checkpoint: {filename}")

    def training_step(self, batch):
        audio = batch['audio']
        texts = batch['text']

        loss, _, _ = self.model(
            text= texts, audio = audio
        )
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)

        should_save = (
            self.global_step > 0
            and self.trainer.is_global_zero
            and self.global_step % self.config.save_interval == 0
        )
        if should_save:
            self.save()

        return loss

    def on_after_backward(self) -> None:
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5
        self.log("train/grad_norm", total_norm, prog_bar=True, sync_dist=True)

    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer) -> None:
        optimizer.zero_grad()
        self.ema_model.update()


def train_model(train_module: TrainModule):
    config = train_module.config
    train_module = train_module
    logger = CometLogger(
            project_name=config.wandb_project,
            experiment_name=config.wandb_run_name or run_name,
        )

    callbacks = [
        LearningRateMonitor(logging_interval="step")

    ]

    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=config.epochs,
        max_steps=config.max_steps,
        accumulate_grad_batches=config.grad_accumulation_steps or 1,
        gradient_clip_val=config.max_grad_norm or 0.0,
        log_every_n_steps=4,
        accelerator="auto",
        devices="auto",
        precision="bf16-mixed" if torch.cuda.is_available() else 32,
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    trainer.fit(
        train_module,
        train_dataloaders=train_module.train_loader,
        val_dataloaders=train_module.valid_loader,
    )
