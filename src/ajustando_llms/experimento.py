"""Compare o treinamento da cabeça de classificação com o ajuste completo."""

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

MODEL = "neuralmind/bert-base-portuguese-cased"
DATA_URL = "https://linguateca.pt/Repositorio/ReLi/"
AUTHORS = {"train": ["Amado", "Meyer", "Orwell", "Salinger", "Sheldon"],
           "validation": ["Reboucas"], "test": ["Saramago"]}
LABELS = {"negative": 0, "positive": 1}


def preparar_dados(tokenizer, max_length, max_train, seed):
    partitions = {}
    for split, authors in AUTHORS.items():
        rows = []
        for author in authors:
            with urlopen(DATA_URL + f"ReLi-{author}.txt", timeout=60) as response:
                text = response.read().decode("utf-8-sig")
            words, polarities = [], set()

            def flush():
                if words and len(polarities) == 1:
                    polarity = next(iter(polarities))
                    rows.append({"sentence": " ".join(words), "label": polarity})
                words.clear()
                polarities.clear()

            for line in text.splitlines():
                if not line.strip() or line.startswith("#"):
                    flush()
                elif not line.startswith("["):
                    fields = line.split("\t")
                    if len(fields) >= 5:
                        words.append(fields[0])
                        if fields[4] == "+":
                            polarities.add("positive")
                        elif fields[4] == "-":
                            polarities.add("negative")
            flush()
        partitions[split] = Dataset.from_list(rows)
    dataset = DatasetDict(partitions)
    if max_train:
        dataset["train"] = dataset["train"].shuffle(seed=seed).select(
            range(min(max_train, len(dataset["train"])))
        )
    counts = {split: dict(Counter(data["label"])) for split, data in dataset.items()}

    def tokenizar(batch):
        encoded = tokenizer(batch["sentence"], truncation=True, max_length=max_length)
        encoded["labels"] = [LABELS[label] for label in batch["label"]]
        return encoded

    dataset = dataset.map(tokenizar, batched=True, remove_columns=dataset["train"].column_names)
    return dataset, counts


def avaliar(model, loader, device):
    model.eval()
    predicted, expected = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            predicted.extend(logits.argmax(dim=-1).cpu().tolist())
            expected.extend(batch["labels"].cpu().tolist())
    return {
        "acuracia": accuracy_score(expected, predicted),
        "f1_macro": f1_score(expected, predicted, labels=[0, 1], average="macro", zero_division=0),
        "matriz_confusao_negativo_positivo": confusion_matrix(expected, predicted, labels=[0, 1]).tolist(),
        "exemplos": len(expected),
    }


def executar(modo, data, tokenizer, args, device):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=2, id2label={0: "negative", 1: "positive"}, label2id=LABELS
    )
    if modo == "cabeca":
        for param in model.base_model.parameters():
            param.requires_grad = False
    model.to(device)
    total = sum(p.numel() for p in model.parameters())
    treinaveis = sum(p.numel() for p in model.parameters() if p.requires_grad)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(data["train"], batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed), collate_fn=collator)
    val_loader = DataLoader(data["validation"], batch_size=args.batch_size, collate_fn=collator)
    test_loader = DataLoader(data["test"], batch_size=args.batch_size, collate_fn=collator)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    historico = []
    inicio = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        validacao = avaliar(model, val_loader, device)
        historico.append({"epoca": epoch + 1, "loss_medio_por_lote": loss_sum / len(train_loader), "validacao": validacao})
        print(f"{modo}: época {epoch + 1}/{args.epochs} — validação: {validacao}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    segundos = time.perf_counter() - inicio
    teste = avaliar(model, test_loader, device)
    return {"modo": modo, "parametros_totais": total, "parametros_treinaveis": treinaveis,
            "segundos_treino_e_validacao": segundos, "historico": historico, "teste": teste,
            "pico_memoria_gpu_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1) if device.type == "cuda" else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modo", choices=["cabeca", "completo", "ambos"], default="ambos")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-train", type=int, default=1000, help="0 usa todos os exemplos de treino")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("resultados/experimento.json"))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 2 or args.max_train < 0 or args.learning_rate <= 0:
        parser.error("Épocas, lote e taxa devem ser positivos; max-length >= 2 e max-train >= 0.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    data, counts = preparar_dados(tokenizer, args.max_length, args.max_train, args.seed)
    print(f"Distribuição: {counts}", flush=True)
    if any(not len(split) for split in data.values()):
        raise ValueError("Uma das partições não tem exemplos positivos ou negativos.")
    modos = ["cabeca", "completo"] if args.modo == "ambos" else [args.modo]
    resultados = []
    for modo in modos:
        resultados.append(executar(modo, data, tokenizer, args, device))
        if device.type == "cuda":
            torch.cuda.empty_cache()
    output = {"modelo": MODEL, "dataset": DATA_URL, "configuracao": {**vars(args), "output": str(args.output)},
              "dispositivo": str(device), "distribuicao": counts, "resultados": resultados}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Resultados salvos em {args.output}")
