import os
import pandas as pd
import numpy as np

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score



MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
CSV_PATH = "desempenho_funcionarios.csv"
OUT_DIR = "./modelo_desempenho_bertimbau"

label2id = {
    "baixo": 0,
    "medio": 1,
    "alto": 2
}
id2label = {v: k for k, v in label2id.items()}



df = pd.read_csv(CSV_PATH)



# Checagens simples
print(df.head())
print("Labels únicos:", df["label"].unique())

# Converte para Dataset do Hugging Face
dataset = Dataset.from_pandas(df)

# Split treino/validação (ex: 80/20)
dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_ds = dataset["train"]
val_ds = dataset["test"]


#  TOKENIZER E PREPROCESSAMENTO

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(examples):
    # Tokeniza o texto
    result = tokenizer(
        examples["texto"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

    result["labels"] = examples["label"]
    return result

train_ds = train_ds.map(preprocess, batched=True)
val_ds = val_ds.map(preprocess, batched=True)


train_ds = train_ds.remove_columns(["texto", "__index_level_0__"]) if "__index_level_0__" in train_ds.column_names else train_ds.remove_columns(["texto"])
val_ds = val_ds.remove_columns(["texto", "__index_level_0__"]) if "__index_level_0__" in val_ds.column_names else val_ds.remove_columns(["texto"])

train_ds.set_format("torch")
val_ds.set_format("torch")



num_labels = len(df["label"].unique())

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id={v: k for k, v in id2label.items()}
)




def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    precision_macro = precision_score(labels, preds, average="macro", zero_division=0)
    recall_macro = recall_score(labels, preds, average="macro", zero_division=0)

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro
    }


# ARGUMENTOS DE TREINO

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Usando device:", device)
model.to(device)

training_args = TrainingArguments(
    output_dir=OUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=50
)



# TRAINER

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

#  TREINO

trainer.train()

# SALVAR MODELO

trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

print("Treino concluído! Modelo salvo em:", OUT_DIR)
