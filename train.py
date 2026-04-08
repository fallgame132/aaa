import argparse
import torch
from config import Config
from data import CharDataset
from model import TinyGPT


def estimate_loss(model: TinyGPT, dataset: CharDataset, cfg: Config, device: str, eval_iters: int = 50) -> dict:
    model.eval()
    losses = {}
    for split in ("train", "val"):
        total = 0.0
        for _ in range(eval_iters):
            x, y = dataset.get_batch(split, cfg.batch_size, device)
            _, loss = model(x, y)
            total += loss.item()
        losses[split] = total / eval_iters
    model.train()
    return losses


def train(max_iters: int | None = None, batch_size: int | None = None):
    cfg = Config()
    if max_iters is not None:
        cfg.max_iters = max_iters
        cfg.eval_interval = max(1, max_iters // 5)
    if batch_size is not None:
        cfg.batch_size = batch_size
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Устройство: {device}")

    dataset = CharDataset(cfg)
    cfg.vocab_size = dataset.vocab_size
    print(f"Словарь: {cfg.vocab_size} символов")
    print(f"Обучающих токенов: {len(dataset.train_data)}")

    model = TinyGPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Параметров модели: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    best_val_loss = float("inf")

    for step in range(1, cfg.max_iters + 1):
        x, y = dataset.get_batch("train", cfg.batch_size, device)
        _, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % cfg.eval_interval == 0 or step == 1:
            losses = estimate_loss(model, dataset, cfg, device)
            print(f"Шаг {step:>5} | train loss: {losses['train']:.4f} | val loss: {losses['val']:.4f}")

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "cfg": cfg,
                        "stoi": dataset.stoi,
                        "itos": dataset.itos,
                    },
                    cfg.checkpoint_path,
                )
                print(f"  -> Сохранён чекпоинт (val loss: {best_val_loss:.4f})")

    print("\nОбучение завершено!")
    print(f"Лучший val loss: {best_val_loss:.4f}")
    print(f"Модель сохранена в: {cfg.checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iters", type=int, default=None, help="Переопределить число шагов обучения")
    parser.add_argument("--batch-size", type=int, default=None, help="Переопределить размер батча")
    args = parser.parse_args()
    train(max_iters=args.max_iters, batch_size=args.batch_size)
