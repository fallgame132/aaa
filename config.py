from dataclasses import dataclass


@dataclass
class Config:
    # Data
    data_path: str = "sample_text.txt"
    block_size: int = 128       # длина контекста (символов)

    # Model
    vocab_size: int = 0         # заполняется при загрузке данных
    n_embd: int = 128           # размерность эмбеддинга
    n_head: int = 4             # число голов внимания
    n_layer: int = 4            # число Transformer-блоков
    dropout: float = 0.1

    # Training
    batch_size: int = 32
    max_iters: int = 3000
    eval_interval: int = 300
    learning_rate: float = 3e-4
    checkpoint_path: str = "checkpoint.pt"

    # Generation
    max_new_tokens: int = 200
    temperature: float = 0.8
