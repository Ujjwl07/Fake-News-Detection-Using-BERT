import os
import re
import random
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    set_seed
)
from torch.optim import AdamW

#Config
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 64
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
SEED = 42

DATA_DIR = "/workspace/fake-news-detector/data"
SAVE_DIR = "/workspace/fake-news-detector/model"

# Setup
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#Load Data
fake_path = os.path.join(DATA_DIR, "Fake.csv")
real_path = os.path.join(DATA_DIR, "True.csv")

fake_df = pd.read_csv(fake_path)
real_df = pd.read_csv(real_path)

fake_df["label"] = 0
real_df["label"] = 1

data = pd.concat([fake_df, real_df], ignore_index=True)

# detect text column
text_cols = ["title", "text", "headline", "article"]
text_col = next((c for c in text_cols if c in data.columns), None)

def clean_text(s):
    if pd.isna(s):
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()

data["text"] = data[text_col].apply(clean_text)
data = data[["text", "label"]]

# Split
train_val_df, test_df = train_test_split(
    data, test_size=0.20, stratify=data["label"], random_state=SEED
)

train_df, val_df = train_test_split(
    train_val_df, test_size=0.20, stratify=train_val_df["label"], random_state=SEED
)

#Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class NewsDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = list(texts)
        self.labels = list(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_loader = DataLoader(NewsDataset(train_df["text"], train_df["label"]), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(NewsDataset(val_df["text"], val_df["label"]), batch_size=BATCH_SIZE)
test_loader  = DataLoader(NewsDataset(test_df["text"], test_df["label"]), batch_size=BATCH_SIZE)

# Model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(device)

optimizer = AdamW(model.parameters(), lr=LR)
total_steps = len(train_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# Train
def train_one_epoch(epoch):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * input_ids.size(0)
        preds = torch.argmax(logits, dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    print(f"Epoch {epoch} Train Loss: {total_loss/total:.4f}, Acc: {correct/total:.4f}")


def eval_model():
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"Validation Acc: {correct/total:.4f}")

# training loop
for epoch in range(1, EPOCHS + 1):
    train_one_epoch(epoch)
    eval_model()

# Test
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch["labels"].numpy())

print("Test Accuracy:", accuracy_score(all_labels, all_preds))
print(classification_report(all_labels, all_preds))

# Save
os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print("Model saved to:", SAVE_DIR)