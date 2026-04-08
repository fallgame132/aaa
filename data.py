import torch
from config import Config


class CharDataset:
    def __init__(self, cfg: Config):
        with open(cfg.data_path, "r", encoding="utf-8") as f:
            text = f.read()

        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        split = int(0.9 * len(data))
        self.train_data = data[:split]
        self.val_data = data[split:]
        self.block_size = cfg.block_size

    def get_batch(self, split: str, batch_size: int, device: str):
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - self.block_size, (batch_size,))
        x = torch.stack([data[i : i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + self.block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([self.stoi[c] for c in text if c in self.stoi], dtype=torch.long)

    def decode(self, tokens: list[int]) -> str:
        return "".join(self.itos[i] for i in tokens)
