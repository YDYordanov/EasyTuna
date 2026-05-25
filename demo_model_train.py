"""
Demo training script with support for both single and multiple constraints.

This script demonstrates:
- Original single constraint support (n_params) for backward compatibility
- New multiple constraint metrics (memory_usage, inference_time, model_flops)
- EasyTuna integration via global variables for optimization target and constraints
"""

import os, argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, BertConfig, BertForSequenceClassification

# Global variables for optimization target and constraints
# EasyTuna will extract these automatically
accuracy = None       # Optimization target (maximize)

# Constraint metrics exposed to EasyTuna
n_params = None       # Model parameter count (legacy constraint)
memory_usage = None   # Peak memory usage in MB
inference_time = None # Average inference time in milliseconds
model_flops = None    # Model computational complexity in FLOPs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_raw = load_dataset("ag_news")
split = _raw["train"].train_test_split(train_size=0.9, seed=42)
train_ds, val_ds = split["train"].select(range(2000)), split["test"].select(range(500))  # Subsample for fast demo runs.
test_ds = _raw["test"]

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess(batch):
    enc = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    enc["labels"] = batch["label"]
    return enc

# Tokenize once, then set PyTorch format.
train_ds = train_ds.map(preprocess, batched=True, remove_columns=["label","text"])
val_ds   = val_ds.map(preprocess,   batched=True, remove_columns=["label","text"])
test_ds  = test_ds.map(preprocess,  batched=True, remove_columns=["label","text"])

for ds in (train_ds, val_ds, test_ds):
    ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])


def get_number_of_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_memory_usage(model, batch_size=32, seq_len=128):
    """Estimate peak memory usage during training (simplified estimation)."""
    # Parameter memory (4 bytes per float32 parameter)
    param_memory = sum(p.numel() * 4 for p in model.parameters()) / (1024**2)  # MB
    
    # Activation memory estimation (rough approximation)
    hidden_size = getattr(model.config, 'hidden_size', 128)
    activation_memory = batch_size * seq_len * hidden_size * 4 / (1024**2)  # MB
    
    # Gradient memory (same as parameters)
    gradient_memory = param_memory
    
    total_memory = param_memory + activation_memory + gradient_memory
    return total_memory


def estimate_model_flops(model, seq_len=128):
    """Estimate computational complexity in FLOPs (simplified estimation)."""
    hidden_size = getattr(model.config, 'hidden_size', 128)
    num_layers = getattr(model.config, 'num_hidden_layers', 1)
    num_heads = getattr(model.config, 'num_attention_heads', 1)
    
    # Rough FLOP estimation for BERT-like transformer
    # Attention: Q*K^T, softmax, attention*V for each head and layer
    attention_flops = num_layers * num_heads * seq_len * seq_len * (hidden_size // num_heads) * 2
    
    # Feed-forward network: 2 linear layers per transformer layer
    intermediate_size = getattr(model.config, 'intermediate_size', hidden_size * 4)
    ffn_flops = num_layers * seq_len * (hidden_size * intermediate_size + intermediate_size * hidden_size)
    
    total_flops = attention_flops + ffn_flops
    return total_flops


def measure_inference_time(model, device, seq_len=128, num_samples=100):
    """Measure average inference time in milliseconds."""
    model.eval()
    dummy_input = torch.randint(0, model.config.vocab_size, (1, seq_len)).to(device)
    dummy_mask = torch.ones(1, seq_len).to(device)
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input, attention_mask=dummy_mask)
    
    # Actual timing
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_samples):
            _ = model(dummy_input, attention_mask=dummy_mask)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    avg_time_ms = (end_time - start_time) * 1000 / num_samples
    return avg_time_ms


"""
1) Build a tiny BERT-for-seq-classification from scratch,
2) Train on AG_NEWS train split,
3) Compute accuracy and n_params.
"""

# ─── parse all hyperparams via CLI ─────────────────────────────────────────────
# Note: This is necessary for EasyTuna
parser = argparse.ArgumentParser(description="Train tiny BERT on AG_NEWS")
parser.add_argument("--lr",               type=float, default=1e-4)
parser.add_argument("--hidden_size",      type=int,   default=128)
parser.add_argument("--num_heads",        type=int,   default=2)
parser.add_argument("--interm_size_ratio",type=int,   default=2)
parser.add_argument("--num_layers",       type=int,   default=3)
parser.add_argument("--dropout_rate",     type=float, default=0.1)
parser.add_argument("--weight_decay",     type=float, default=0.01)
parser.add_argument("--batch_size",       type=int,   default=32)
parser.add_argument("--epochs",           type=int,   default=3)
parser.add_argument("--seed",             type=int,   default=42)
args = parser.parse_args()

# Bind CLI args to local names for readability.
lr               = args.lr
hidden_size      = args.hidden_size
num_heads        = args.num_heads
interm_size_ratio= args.interm_size_ratio
num_layers       = args.num_layers
dropout_rate     = args.dropout_rate
weight_decay     = args.weight_decay
batch_size       = args.batch_size
epochs           = args.epochs
seed             = args.seed
LOG_DIR   = os.environ.get('LOG_DIR')  # Trial output directory provided by EasyTuna.

# Set random seeds for reproducibility.
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- build model config ---
config = BertConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=hidden_size,
    num_hidden_layers=num_layers,
    num_attention_heads=num_heads,  # Must divide hidden_size.
    intermediate_size=hidden_size * interm_size_ratio,
    hidden_dropout_prob=dropout_rate,
    attention_probs_dropout_prob=dropout_rate,
    num_labels=4
)
model = BertForSequenceClassification(config).to(device)

# --- data loaders ---
train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True
)
val_loader = DataLoader(
    val_ds, batch_size=batch_size, shuffle=False
)

# --- optimizer & loss ---
optimizer = optim.AdamW(
    model.parameters(),
    lr=lr,
    weight_decay=weight_decay
)
criterion = nn.CrossEntropyLoss()

# --- train ---
model.train()
for _ in range(epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        mask      = batch["attention_mask"].to(device)
        labels    = batch["labels"].to(device)
        outputs   = model(input_ids, attention_mask=mask).logits
        loss      = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# --- validate ---
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        mask      = batch["attention_mask"].to(device)
        labels    = batch["labels"].to(device)
        preds     = model(input_ids, attention_mask=mask).logits.argmax(dim=-1)
        correct  += (preds == labels).sum().item()
        total    += labels.size(0)

# Set target metrics as global variables (REQUIRED for EasyTuna)
accuracy = correct / total
n_params = get_number_of_parameters(model)

# Calculate additional constraint metrics
memory_usage = estimate_memory_usage(model, batch_size=batch_size)
model_flops = estimate_model_flops(model)
inference_time = measure_inference_time(model, device)

# Print for monitoring (optional)
print(f"accuracy={accuracy}")
print(f"n_params={n_params}")
print(f"memory_usage={memory_usage:.1f} MB")
print(f"model_flops={model_flops:,}")
print(f"inference_time={inference_time:.2f} ms")

# Optional: Save model if LOG_DIR is available
if LOG_DIR:
    model_path = os.path.join(LOG_DIR, "model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")
