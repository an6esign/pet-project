# src/serving/cli.py
import argparse
from typing import List
from ..models.infer import load_router

def run_interactive() -> None:
    infer = load_router()
    print("🔮 Ticket Router (TF-IDF). Введите текст (или 'exit' для выхода).")
    while True:
        try:
            line = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Выход.")
            break
        if not line or line.lower() in {"exit", "quit"}:
            print("👋 Выход.")
            break
        pred = infer(line)[0]
        proba_str = f" | proba={pred.proba}" if pred.proba is not None else ""
        print(f"[{pred.label_id} | {pred.label_name}]{proba_str}")

def run_file(path: str) -> None:
    infer = load_router()
    with open(path, "r", encoding="utf-8") as f:
        texts: List[str] = [line.strip() for line in f if line.strip()]
    preds = infer(texts)
    for t, p in zip(texts, preds):
        proba_str = f" | proba={p.proba}" if p.proba is not None else ""
        print(f"[{p.label_id} | {p.label_name}]{proba_str} -> {t}")

def main():
    parser = argparse.ArgumentParser(description="CLI для инференса Ticket Router")
    parser.add_argument("--file", "-f", type=str, default=None,
                        help="Путь к txt-файлу (по строке на запрос). Если не указан — интерактивный режим.")
    args = parser.parse_args()
    if args.file:
        run_file(args.file)
    else:
        run_interactive()

if __name__ == "__main__":
    main()
