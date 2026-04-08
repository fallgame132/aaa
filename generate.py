#!/usr/bin/env python3
"""
CLI для генерации текста с помощью обученного Tiny GPT.

Использование:
    python generate.py
    python generate.py --prompt "Привет" --tokens 300 --temperature 0.8
    python generate.py --checkpoint checkpoint.pt
"""
import argparse
import torch
from config import Config
from model import TinyGPT


def load_model(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg: Config = ckpt["cfg"]
    stoi: dict = ckpt["stoi"]
    itos: dict = ckpt["itos"]

    model = TinyGPT(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg, stoi, itos


def generate(model: TinyGPT, prompt: str, stoi: dict, itos: dict,
             max_new_tokens: int, temperature: float, device: str) -> str:
    tokens = [stoi[c] for c in prompt if c in stoi]
    if not tokens:
        tokens = [0]
    idx = torch.tensor([tokens], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature)
    return "".join(itos[i] for i in out[0].tolist())


def interactive(model: TinyGPT, cfg: Config, stoi: dict, itos: dict, device: str) -> None:
    print("=" * 50)
    print("  Mini AI — Tiny GPT  |  Генерация текста")
    print("  Введите 'exit' для выхода")
    print("=" * 50)
    while True:
        try:
            prompt = input("\nПромпт > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            break
        if prompt.lower() in ("exit", "quit", "выход"):
            print("До свидания!")
            break
        if not prompt:
            continue
        result = generate(model, prompt, stoi, itos, cfg.max_new_tokens, cfg.temperature, device)
        print("\n" + "-" * 40)
        print(result)
        print("-" * 40)


def main():
    parser = argparse.ArgumentParser(description="Tiny GPT — генерация текста")
    parser.add_argument("--checkpoint", default="checkpoint.pt", help="Путь к чекпоинту")
    parser.add_argument("--prompt", default=None, help="Начальный текст (если не указан — интерактивный режим)")
    parser.add_argument("--tokens", type=int, default=None, help="Количество генерируемых токенов")
    parser.add_argument("--temperature", type=float, default=None, help="Температура (0.1–2.0)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        model, cfg, stoi, itos = load_model(args.checkpoint, device)
    except FileNotFoundError:
        print(f"Чекпоинт '{args.checkpoint}' не найден.")
        print("Сначала обучите модель: python train.py")
        return

    if args.tokens is not None:
        cfg.max_new_tokens = args.tokens
    if args.temperature is not None:
        cfg.temperature = args.temperature

    if args.prompt is not None:
        result = generate(model, args.prompt, stoi, itos, cfg.max_new_tokens, cfg.temperature, device)
        print(result)
    else:
        interactive(model, cfg, stoi, itos, device)


if __name__ == "__main__":
    main()
