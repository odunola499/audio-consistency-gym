import string
from typing import List, Literal, Optional, Union

import torch
from torch import LongTensor
from torch.nn.utils.rnn import pad_sequence


class CharTokenizer:
    def __init__(self, vocab_file: Optional[str] = None):

        if vocab_file is not None:
            with open(vocab_file) as fp:
                chars = fp.readlines()
                chars = [i.strip() for i in chars]
        else:
            chars = (
                string.ascii_letters
                + string.digits
                + string.punctuation
                + string.whitespace
            )

        self.ids_to_chars = {index + 1: char for index, char in enumerate(chars)}
        self.chars_to_ids = {j: i for i, j in self.ids_to_chars.items()}
        self.unk_id = len(self.chars_to_ids)
        self.vocab_size = len(self.ids_to_chars)

    def encode(self, text: str):
        return [self.chars_to_ids.get(i, self.unk_id) for i in text]

    def batch_encode(self, texts: List[str]):
        return [self.encode(text) for text in texts]

    def decode(self, input_ids: Union[List[int], LongTensor]):
        if isinstance(input_ids, LongTensor):
            input_ids = input_ids.tolist()
        return [self.ids_to_chars[i] for i in input_ids]

    def batch_decode(self, inputs_ids: Union[List[List[int]], LongTensor]):
        if isinstance(inputs_ids, LongTensor):
            inputs_ids = inputs_ids.tolist()
        return [self.decode(i) for i in inputs_ids]

    def __call__(
        self,
        texts: Union[List[str], str],
        padding: Literal["longest", "max_length"] = "longest",
        truncation: bool = True,
        max_length=512,
        padding_value=0,
        device: Optional = None,
        return_attention_mask: bool = True,
    ):
        if isinstance(texts, str):
            texts = [texts]

        inputs_ids = self.batch_encode(texts)
        inputs_ids = [
            torch.tensor(input_ids, dtype=torch.long, device=device)
            for input_ids in inputs_ids
        ]

        inputs_ids = pad_sequence(
            inputs_ids, batch_first=True, padding_value=padding_value
        )
        seq_length = inputs_ids.shape[-1]

        if truncation:
            inputs_ids = inputs_ids[:, :max_length]
        if padding == "max_length":
            if seq_length < max_length:
                remaining_pad = max_length - seq_length
                inputs_ids = torch.nn.functional.pad(inputs_ids, (0, remaining_pad))
            else:
                inputs_ids = inputs_ids[:, :max_length]
        else:
            assert (
                padding == "longest"
            ), "Only 'max_length' and 'longest' is allowed for padding args"
        if return_attention_mask:
            attention_mask = (inputs_ids != 0).to(torch.long)
            return inputs_ids, attention_mask
        return inputs_ids, None

    def save(self, file="vocab.txt", chars=None):
        if chars is None:
            chars = self.chars_to_ids.keys()
        with open(file, "w") as fp:
            for char in chars:
                fp.write(char)
                fp.write("\n")

    @classmethod
    def load(cls, file: str):
        return cls(vocab_file=file)


if __name__ == "__main__":
    tokenizer = CharTokenizer()
    texts = ["my name is odunola", "I am a boy."]
    output, _ = tokenizer(texts)
    print(output.shape)
    print(tokenizer(texts))
    print(tokenizer.unk_id)
    print(tokenizer.vocab_size)

    tokenizer.save()
