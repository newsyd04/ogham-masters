#!/usr/bin/env python3
"""Append Phase 2 (synth-freeform fine-tuning + test eval) cells to the
CNN+RNN and PARSeq baseline notebooks. Idempotent — re-running overwrites
the appended cells if they exist.
"""

import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
NB_DIR = PROJECT / "notebooks"

PHASE2_MARKER = "### PHASE 2: Fine-tune on synthetic-freeform"


def make_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True) if isinstance(source, str) else source,
    }


def make_code_cell(source):
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": source.splitlines(keepends=True) if isinstance(source, str) else source,
    }


# ==========================================================================
# CNN+RNN Phase 2 cells
# ==========================================================================
CNN_RNN_PHASE2_MD = """---

### PHASE 2: Fine-tune on synthetic-freeform

Fine-tune the Phase 1 CNN+RNN checkpoint on 280 synthetic-freeform samples
and evaluate on a held-out test split of 35 samples.
"""

CNN_RNN_PHASE2_SETUP = """# ============================================================
# PHASE 2: Load synthetic-freeform data
# ============================================================
import csv, random, shutil
from pathlib import Path

FREEFORM_DRIVE = Path('/content/drive/MyDrive/ogham_ocr/datasets/synthetic_freeform')
LOCAL_FF = Path(REPO_DIR) / 'ogham_dataset' / 'synthetic_freeform'
P2_CHECKPOINT_DIR = f'{DRIVE_ROOT}/checkpoints/cnn_rnn_phase2'
os.makedirs(P2_CHECKPOINT_DIR, exist_ok=True)

P2_EPOCHS = 5
P2_LR = 1e-4
P2_BATCH_SIZE = 32

LOCAL_FF.mkdir(parents=True, exist_ok=True)
!rsync -a {FREEFORM_DRIVE}/ {LOCAL_FF}/

# Deterministic 80/10/10 split (seed 42 for reproducibility with TrOCR experiments)
freeform_labels = {}
all_ids = []
with open(LOCAL_FF / 'labels.csv', newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        idx = int(row['image_file'].replace('freeform_', '').replace('.png', ''))
        freeform_labels[idx] = (row['image_file'], row['ogham_text'])
        all_ids.append(idx)

random.seed(42)
random.shuffle(all_ids)
n = len(all_ids)
n_val = n_test = max(1, n // 10)
n_train = n - n_val - n_test
p2_train_ids = all_ids[:n_train]
p2_val_ids   = all_ids[n_train:n_train + n_val]
p2_test_ids  = all_ids[n_train + n_val:]
print(f'Freeform split: train={len(p2_train_ids)}, val={len(p2_val_ids)}, test={len(p2_test_ids)}')


class FreeformCTCDataset(torch.utils.data.Dataset):
    def __init__(self, indices, transform, labels_map, char_to_idx, max_label_len=64):
        self.indices = indices
        self.transform = transform
        self.labels_map = labels_map
        self.char_to_idx = char_to_idx
        self.max_label_len = max_label_len
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, i):
        idx = self.indices[i]
        fname, text = self.labels_map[idx]
        img = Image.open(LOCAL_FF / 'images' / fname).convert('RGB')
        img = self.transform(img)
        text = text.replace('\\u1680', ' ')
        label = [self.char_to_idx.get(c, 0) for c in text if c in self.char_to_idx]
        label_length = len(label)
        label = label[:self.max_label_len]
        label_length = min(label_length, self.max_label_len)
        label = label + [0] * (self.max_label_len - len(label))
        return {
            'image': img,
            'label': torch.tensor(label, dtype=torch.long),
            'label_length': torch.tensor(label_length, dtype=torch.long),
            'text': text,
        }


p2_train_loader = DataLoader(
    FreeformCTCDataset(p2_train_ids, train_transform, freeform_labels, char_to_idx),
    batch_size=P2_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
)
p2_val_loader = DataLoader(
    FreeformCTCDataset(p2_val_ids, val_transform, freeform_labels, char_to_idx),
    batch_size=P2_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
)
p2_test_loader = DataLoader(
    FreeformCTCDataset(p2_test_ids, val_transform, freeform_labels, char_to_idx),
    batch_size=P2_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
)
print('Phase 2 data loaders ready')
"""

CNN_RNN_PHASE2_TRAIN = """# ============================================================
# PHASE 2: Load best Phase 1 checkpoint and fine-tune
# ============================================================
# Seed Phase 2 checkpoint dir from Phase 1 best
p1_ckpt = f'{CHECKPOINT_DIR}/best_model.pt'
p2_ckpt = f'{P2_CHECKPOINT_DIR}/best_model.pt'
if not os.path.exists(p2_ckpt) and os.path.exists(p1_ckpt):
    shutil.copyfile(p1_ckpt, p2_ckpt)
    print(f'Seeded Phase 2 ckpt from Phase 1 best')

# Load checkpoint weights into model
state = torch.load(p2_ckpt, map_location=device)
model.load_state_dict(state)
print('Loaded Phase 1 weights into model')

# Lower-LR fine-tune on freeform only (pure fine-tune, no curriculum mix)
p2_optimizer = torch.optim.Adam(model.parameters(), lr=P2_LR)

p2_history = {'train_loss': [], 'val_cer': [], 'val_exact_match': []}
p2_best_cer = float('inf')

print(f'\\n{"="*60}')
print(f'Phase 2 CNN+RNN fine-tune: {P2_EPOCHS} epochs, batch {P2_BATCH_SIZE}, lr {P2_LR}')
print(f'{"="*60}\\n')

for epoch in range(P2_EPOCHS):
    epoch_start = time.time()
    model.train()
    train_loss, n_batches = 0, 0
    for batch in tqdm(p2_train_loader, desc=f'  Epoch {epoch+1}/{P2_EPOCHS}', leave=False):
        images = batch['image'].to(device)
        labels = batch['label']
        label_lengths = batch['label_length']
        p2_optimizer.zero_grad()
        logits = model(images)          # (B, T, num_classes)
        log_probs = logits.log_softmax(2).transpose(0, 1)  # (T, B, C) for CTC
        input_lengths = torch.full((images.size(0),), logits.size(1), dtype=torch.long)
        loss = ctc_loss_fn(log_probs, labels, input_lengths, label_lengths)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        p2_optimizer.step()
        train_loss += loss.item()
        n_batches += 1
    avg_train_loss = train_loss / max(n_batches, 1)

    # Val (character-level CER, space-stripped)
    model.eval()
    refs, preds = [], []
    with torch.no_grad():
        for batch in p2_val_loader:
            logits = model(batch['image'].to(device))
            preds += ctc_decode_greedy(logits.cpu(), idx_to_char, BLANK_IDX)
            refs += list(batch['text'])
    refs_ns = [r.replace(' ', '') for r in refs]
    preds_ns = [p.replace(' ', '') for p in preds]
    val_cer = compute_cer(preds_ns, refs_ns)
    val_exact = sum(1 for p, r in zip(preds_ns, refs_ns) if p == r) / max(len(refs_ns), 1)

    p2_history['train_loss'].append(avg_train_loss)
    p2_history['val_cer'].append(val_cer)
    p2_history['val_exact_match'].append(val_exact)

    if val_cer < p2_best_cer:
        p2_best_cer = val_cer
        torch.save(model.state_dict(), p2_ckpt)

    print(f'Epoch {epoch+1}/{P2_EPOCHS} ({time.time()-epoch_start:.0f}s) | '
          f'Train Loss: {avg_train_loss:.4f} | Val CER (no-sp): {val_cer*100:.2f}% | '
          f'Val Exact: {val_exact*100:.1f}%')

with open(f'{P2_CHECKPOINT_DIR}/history.json', 'w') as f:
    json.dump(p2_history, f, indent=2)
print(f'\\nBest Phase 2 val CER: {p2_best_cer*100:.2f}%')
"""

CNN_RNN_PHASE2_EVAL = """# ============================================================
# PHASE 2: Evaluate on pristine test split
# ============================================================
# Load best Phase 2 checkpoint
model.load_state_dict(torch.load(p2_ckpt, map_location=device))
model.eval()

refs, preds = [], []
with torch.no_grad():
    for batch in p2_test_loader:
        logits = model(batch['image'].to(device))
        preds += ctc_decode_greedy(logits.cpu(), idx_to_char, BLANK_IDX)
        refs += list(batch['text'])

# Character-level metrics (space-stripped)
refs_ns = [r.replace(' ', '') for r in refs]
preds_ns = [p.replace(' ', '') for p in preds]
test_cer = compute_cer(preds_ns, refs_ns)
test_exact = sum(1 for p, r in zip(preds_ns, refs_ns) if p == r) / max(len(refs_ns), 1)

print(f'\\n{"="*70}')
print(f'CNN+RNN Phase 2 — TEST SPLIT ({len(refs)} pristine samples)')
print(f'{"="*70}')
print(f'  Character-level CER: {test_cer*100:.2f}%')
print(f'  Exact match:         {test_exact*100:.1f}% ({sum(1 for p,r in zip(preds_ns, refs_ns) if p==r)}/{len(refs_ns)})')
print(f'{"="*70}')

# Save results
p2_results = {
    'model': 'CNN+RNN (ResNet18 + BiLSTM + CTC) — Phase 2',
    'training': {'epochs': P2_EPOCHS, 'lr': P2_LR, 'batch_size': P2_BATCH_SIZE,
                 'train_samples': len(p2_train_ids), 'val_samples': len(p2_val_ids)},
    'test': {'n_samples': len(refs), 'cer_pct': test_cer * 100,
             'exact_match_pct': test_exact * 100,
             'metric_note': 'CER and exact match computed on Ogham character sequences with whitespace excluded'},
    'history': p2_history,
    'per_sample': [{'ref': r, 'pred': p} for r, p in zip(refs, preds)],
}
with open(f'{P2_CHECKPOINT_DIR}/phase2_test_results.json', 'w') as f:
    json.dump(p2_results, f, indent=2, ensure_ascii=False)
print(f'\\nResults saved to {P2_CHECKPOINT_DIR}/phase2_test_results.json')
"""


# ==========================================================================
# PARSeq Phase 2 cells
# ==========================================================================
PARSEQ_PHASE2_MD = """---

### PHASE 2: Fine-tune on synthetic-freeform

Fine-tune the Phase 1 PARSeq checkpoint on 280 synthetic-freeform samples
and evaluate on a held-out test split of 35 samples.
"""

PARSEQ_PHASE2_SETUP = """# ============================================================
# PHASE 2: Load synthetic-freeform data
# ============================================================
import csv, random, shutil
from pathlib import Path

FREEFORM_DRIVE = Path('/content/drive/MyDrive/ogham_ocr/datasets/synthetic_freeform')
LOCAL_FF = Path(REPO_DIR) / 'ogham_dataset' / 'synthetic_freeform'
P2_CHECKPOINT_DIR = f'{DRIVE_ROOT}/checkpoints/parseq_phase2'
os.makedirs(P2_CHECKPOINT_DIR, exist_ok=True)

P2_EPOCHS = 5
P2_LR = 1e-5
P2_BATCH_SIZE = 32

LOCAL_FF.mkdir(parents=True, exist_ok=True)
!rsync -a {FREEFORM_DRIVE}/ {LOCAL_FF}/

# Deterministic 80/10/10 split (same seed 42 as other Phase 2 experiments)
freeform_labels = {}
all_ids = []
with open(LOCAL_FF / 'labels.csv', newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        idx = int(row['image_file'].replace('freeform_', '').replace('.png', ''))
        freeform_labels[idx] = (row['image_file'], row['ogham_text'])
        all_ids.append(idx)

random.seed(42)
random.shuffle(all_ids)
n = len(all_ids)
n_val = n_test = max(1, n // 10)
n_train = n - n_val - n_test
p2_train_ids = all_ids[:n_train]
p2_val_ids   = all_ids[n_train:n_train + n_val]
p2_test_ids  = all_ids[n_train + n_val:]
print(f'Freeform split: train={len(p2_train_ids)}, val={len(p2_val_ids)}, test={len(p2_test_ids)}')


class FreeformPARSeqDataset(torch.utils.data.Dataset):
    def __init__(self, indices, transform, labels_map, max_label_len=25):
        self.indices = indices
        self.transform = transform
        self.labels_map = labels_map
        self.max_label_len = max_label_len
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, i):
        idx = self.indices[i]
        fname, text = self.labels_map[idx]
        img = Image.open(LOCAL_FF / 'images' / fname).convert('RGB')
        img = self.transform(img)
        text = text.replace('\\u1680', ' ')[:self.max_label_len]
        return img, text


p2_train_loader = DataLoader(
    FreeformPARSeqDataset(p2_train_ids, train_transform, freeform_labels, MAX_LABEL_LEN),
    batch_size=P2_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
)
p2_val_loader = DataLoader(
    FreeformPARSeqDataset(p2_val_ids, val_transform, freeform_labels, MAX_LABEL_LEN),
    batch_size=P2_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
)
p2_test_loader = DataLoader(
    FreeformPARSeqDataset(p2_test_ids, val_transform, freeform_labels, MAX_LABEL_LEN),
    batch_size=P2_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
)
print('Phase 2 data loaders ready')
"""

PARSEQ_PHASE2_TRAIN = """# ============================================================
# PHASE 2: Load best Phase 1 checkpoint and fine-tune
# ============================================================
p1_ckpt = f'{CHECKPOINT_DIR}/best_model.pt'
p2_ckpt = f'{P2_CHECKPOINT_DIR}/best_model.pt'
if not os.path.exists(p2_ckpt) and os.path.exists(p1_ckpt):
    shutil.copyfile(p1_ckpt, p2_ckpt)
    print(f'Seeded Phase 2 ckpt from Phase 1 best')

state = torch.load(p2_ckpt, map_location=device)
model.load_state_dict(state)
print('Loaded Phase 1 weights into model')

p2_optimizer = torch.optim.AdamW(model.parameters(), lr=P2_LR, weight_decay=0.01)
p2_history = {'train_loss': [], 'val_cer': [], 'val_exact_match': []}
p2_best_cer = float('inf')

print(f'\\n{"="*60}')
print(f'Phase 2 PARSeq fine-tune: {P2_EPOCHS} epochs, batch {P2_BATCH_SIZE}, lr {P2_LR}')
print(f'{"="*60}\\n')

for epoch in range(P2_EPOCHS):
    epoch_start = time.time()
    model.train()
    train_loss, n_batches = 0, 0
    for images, texts in tqdm(p2_train_loader, desc=f'  Epoch {epoch+1}/{P2_EPOCHS}', leave=False):
        images = images.to(device)
        targets = model.tokenizer.encode(texts, device=device)
        p2_optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            loss = model.forward_logits_loss(images, targets)[1]
        if not torch.isfinite(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        p2_optimizer.step()
        train_loss += loss.item()
        n_batches += 1
    avg_train_loss = train_loss / max(n_batches, 1)

    # Val (character-level, space-stripped)
    model.eval()
    refs, preds = [], []
    with torch.no_grad():
        for images, texts in p2_val_loader:
            images = images.to(device)
            logits = model(images, max_length=MAX_LABEL_LEN)
            decoded = model.tokenizer.decode(logits)
            preds += [p[0] if isinstance(p, tuple) else p for p in decoded]
            refs += list(texts)
    refs_ns = [r.replace(' ', '') for r in refs]
    preds_ns = [p.replace(' ', '') for p in preds]
    val_cer = compute_cer(preds_ns, refs_ns)
    val_exact = sum(1 for p, r in zip(preds_ns, refs_ns) if p == r) / max(len(refs_ns), 1)

    p2_history['train_loss'].append(avg_train_loss)
    p2_history['val_cer'].append(val_cer)
    p2_history['val_exact_match'].append(val_exact)

    if val_cer < p2_best_cer:
        p2_best_cer = val_cer
        torch.save(model.state_dict(), p2_ckpt)

    print(f'Epoch {epoch+1}/{P2_EPOCHS} ({time.time()-epoch_start:.0f}s) | '
          f'Train Loss: {avg_train_loss:.4f} | Val CER (no-sp): {val_cer*100:.2f}% | '
          f'Val Exact: {val_exact*100:.1f}%')

with open(f'{P2_CHECKPOINT_DIR}/history.json', 'w') as f:
    json.dump(p2_history, f, indent=2)
print(f'\\nBest Phase 2 val CER: {p2_best_cer*100:.2f}%')
"""

PARSEQ_PHASE2_EVAL = """# ============================================================
# PHASE 2: Evaluate on pristine test split
# ============================================================
model.load_state_dict(torch.load(p2_ckpt, map_location=device))
model.eval()

refs, preds = [], []
with torch.no_grad():
    for images, texts in p2_test_loader:
        images = images.to(device)
        logits = model(images, max_length=MAX_LABEL_LEN)
        decoded = model.tokenizer.decode(logits)
        preds += [p[0] if isinstance(p, tuple) else p for p in decoded]
        refs += list(texts)

refs_ns = [r.replace(' ', '') for r in refs]
preds_ns = [p.replace(' ', '') for p in preds]
test_cer = compute_cer(preds_ns, refs_ns)
test_exact = sum(1 for p, r in zip(preds_ns, refs_ns) if p == r) / max(len(refs_ns), 1)

print(f'\\n{"="*70}')
print(f'PARSeq Phase 2 — TEST SPLIT ({len(refs)} pristine samples)')
print(f'{"="*70}')
print(f'  Character-level CER: {test_cer*100:.2f}%')
print(f'  Exact match:         {test_exact*100:.1f}% ({sum(1 for p,r in zip(preds_ns, refs_ns) if p==r)}/{len(refs_ns)})')
print(f'{"="*70}')

p2_results = {
    'model': 'PARSeq (ViT + Transformer decoder) — Phase 2',
    'training': {'epochs': P2_EPOCHS, 'lr': P2_LR, 'batch_size': P2_BATCH_SIZE,
                 'train_samples': len(p2_train_ids), 'val_samples': len(p2_val_ids)},
    'test': {'n_samples': len(refs), 'cer_pct': test_cer * 100,
             'exact_match_pct': test_exact * 100,
             'metric_note': 'CER and exact match computed on Ogham character sequences with whitespace excluded'},
    'history': p2_history,
    'per_sample': [{'ref': r, 'pred': p} for r, p in zip(refs, preds)],
}
with open(f'{P2_CHECKPOINT_DIR}/phase2_test_results.json', 'w') as f:
    json.dump(p2_results, f, indent=2, ensure_ascii=False)
print(f'\\nResults saved to {P2_CHECKPOINT_DIR}/phase2_test_results.json')
"""


def strip_existing_phase2(nb):
    """Remove any existing Phase 2 cells added by this script (for idempotency)."""
    cells = nb["cells"]
    # Find index of the Phase 2 marker cell (if any)
    for i, c in enumerate(cells):
        src = "".join(c.get("source", []))
        if PHASE2_MARKER in src:
            nb["cells"] = cells[:i]
            return True
    return False


def append_cells(nb_path, cells):
    with open(nb_path) as f:
        nb = json.load(f)
    strip_existing_phase2(nb)
    nb["cells"].extend(cells)
    with open(nb_path, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)


cnn_rnn_cells = [
    make_markdown_cell(CNN_RNN_PHASE2_MD),
    make_code_cell(CNN_RNN_PHASE2_SETUP),
    make_code_cell(CNN_RNN_PHASE2_TRAIN),
    make_code_cell(CNN_RNN_PHASE2_EVAL),
]
parseq_cells = [
    make_markdown_cell(PARSEQ_PHASE2_MD),
    make_code_cell(PARSEQ_PHASE2_SETUP),
    make_code_cell(PARSEQ_PHASE2_TRAIN),
    make_code_cell(PARSEQ_PHASE2_EVAL),
]

append_cells(NB_DIR / "06_cnn_rnn_baseline.ipynb", cnn_rnn_cells)
print(f"Added {len(cnn_rnn_cells)} Phase 2 cells to 06_cnn_rnn_baseline.ipynb")

append_cells(NB_DIR / "07_parseq_baseline.ipynb", parseq_cells)
print(f"Added {len(parseq_cells)} Phase 2 cells to 07_parseq_baseline.ipynb")
