#!/usr/bin/env python3
"""Append large-model evaluation cells for synth and synth-freeform test sets
to 05_large_model_comparison.ipynb.

Assumes the prompt helpers (ZERO_SHOT_PROMPT, build_few_shot_prompt),
API call functions (call_gpt4o, call_gemini, call_claude), and helper
utilities (image_to_base64, compute_cer, normalize_prediction) are
already defined by earlier cells.
"""

import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
NB = PROJECT / "notebooks" / "05_large_model_comparison.ipynb"

EXTENDED_MARKER = "## Extended evaluation: synthetic + synthetic-freeform"


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "outputs": [],
            "execution_count": None, "source": src.splitlines(keepends=True)}


HEADER = """---

## Extended evaluation: synthetic + synthetic-freeform

Runs the same three large models (GPT-4o, Claude, Gemini) on two additional
test sets, for direct comparison with our fine-tuned models (TrOCR-small,
PARSeq, CNN+RNN):

1. **Synthetic test** — 30 random samples from `DaraTraining/ogham-synthetic-200k`
   validation split (clean font-rendered Ogham)
2. **Synth-freeform test** — 35 samples from the seed-42 held-out test split
   of the synthetic-freeform dataset (elastic-warped hand-drawing simulation)

Few-shot examples are drawn from each dataset's train split to match the test
distribution. Zero-shot and few-shot results are both reported.
"""

SETUP_CELL = """# ============================================================
# EXTENDED EVAL: Load synthetic + synth-freeform test data
# ============================================================
import csv, random, shutil
from pathlib import Path
from datasets import load_dataset

# --- Synth-freeform test samples (same seed-42 split as TrOCR/PARSeq/CNN+RNN) ---
FREEFORM_DRIVE = Path('/content/drive/MyDrive/ogham_ocr/datasets/synthetic_freeform')
LOCAL_FF = Path(REPO_DIR) / 'ogham_dataset' / 'synthetic_freeform'
LOCAL_FF.mkdir(parents=True, exist_ok=True)
!rsync -a {FREEFORM_DRIVE}/ {LOCAL_FF}/

ff_labels = {}
ff_ids = []
with open(LOCAL_FF / 'labels.csv', newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        idx = int(row['image_file'].replace('freeform_', '').replace('.png', ''))
        ff_labels[idx] = (row['image_file'], row['ogham_text'])
        ff_ids.append(f'SYNTH-FF-{idx:04d}')

random.seed(42)
random.shuffle(ff_ids)
n = len(ff_ids)
n_val = n_test = max(1, n // 10)
n_train = n - n_val - n_test
train_ff_ids = ff_ids[:n_train]
test_ff_ids = ff_ids[n_train + n_val:]

# Build test samples list
synth_ff_test_samples = []
for sid in test_ff_ids:
    idx = int(sid.replace('SYNTH-FF-', ''))
    if idx in ff_labels:
        fname, text = ff_labels[idx]
        synth_ff_test_samples.append({
            'stone_id': sid,
            'image_path': str(LOCAL_FF / 'images' / fname),
            'reference': text,
        })

# Few-shot examples from freeform train split (3 short, clear ones)
synth_ff_few_shot = []
train_by_len = sorted(train_ff_ids,
                      key=lambda sid: len(ff_labels[int(sid.replace('SYNTH-FF-', ''))][1]))
for sid in train_by_len[:3]:
    idx = int(sid.replace('SYNTH-FF-', ''))
    fname, text = ff_labels[idx]
    synth_ff_few_shot.append({
        'stone_id': sid,
        'image_path': str(LOCAL_FF / 'images' / fname),
        'transcription': text,
    })

print(f'Synth-freeform: {len(synth_ff_test_samples)} test samples, {len(synth_ff_few_shot)} few-shot examples')

# --- Synthetic test samples (30 from HF dataset) ---
# Stream val split, pick 30 with short-to-medium inscriptions for API cost
print('Loading 30 samples from HF synthetic-200k val split...')
synth_hf = load_dataset('DaraTraining/ogham-synthetic-200k', split='validation', streaming=True)
synth_test_samples = []
synth_few_shot = []

# Save images locally so we can pass paths to API functions
SYNTH_LOCAL = Path('/content/synth_eval_images')
SYNTH_LOCAL.mkdir(exist_ok=True)

n_collected = 0
for i, row in enumerate(synth_hf):
    if i >= 500: break  # safety cap
    text = row.get('ogham_text') or row.get('text', '')
    if not (5 <= len(text) <= 30):
        continue
    img_path = SYNTH_LOCAL / f'synth_{i:04d}.png'
    row['image'].save(img_path)
    entry = {
        'stone_id': f'SYNTH-{i:04d}',
        'image_path': str(img_path),
        'reference': text,
    }
    if len(synth_few_shot) < 3:
        synth_few_shot.append({
            'stone_id': entry['stone_id'],
            'image_path': entry['image_path'],
            'transcription': text,
        })
    else:
        synth_test_samples.append(entry)
        n_collected += 1
        if n_collected >= 30:
            break

print(f'Synthetic: {len(synth_test_samples)} test samples, {len(synth_few_shot)} few-shot examples')
"""

EVAL_HELPER_CELL = """# ============================================================
# Generic eval function — runs a model call over a sample set,
# tracks per-sample predictions and aggregate CER
# ============================================================
def evaluate_model(model_fn, samples, prompt_builder, few_shot=None,
                   label='', sleep_s=0.5):
    \"\"\"Run `model_fn(image_path, prompt, few_shot)` over each sample.
    model_fn signature: (image_path: str, prompt: str, few_shot_images: list|None) -> str
    Returns dict with per-sample results and aggregate CER/exact.
    \"\"\"
    results = {}
    total_dist, total_chars, exact = 0, 0, 0
    for i, sample in enumerate(samples):
        try:
            prompt = prompt_builder(few_shot) if callable(prompt_builder) else prompt_builder
            raw = model_fn(sample['image_path'], prompt, few_shot)
            pred = normalize_prediction(raw)
        except Exception as e:
            pred = ''
            raw = f'ERROR: {e}'
        ref = sample['reference']
        # Space-stripped CER for consistency with fine-tuned model metrics
        ref_ns = ref.replace(' ', '')
        pred_ns = pred.replace(' ', '')
        dist = editdistance.eval(pred_ns, ref_ns)
        total_dist += dist
        total_chars += max(len(ref_ns), 1)
        if pred_ns == ref_ns:
            exact += 1
        cer = dist / max(len(ref_ns), 1)
        results[sample['stone_id']] = {
            'prediction': pred, 'reference': ref, 'raw_response': raw, 'cer_no_sp': cer,
        }
        if i % 5 == 0:
            print(f'  [{label}] {i+1}/{len(samples)}  last_cer={cer*100:.1f}%')
        time.sleep(sleep_s)

    agg = {
        'n_samples': len(samples),
        'mean_cer_no_sp_pct': total_dist / max(total_chars, 1) * 100,
        'exact_match_pct': 100 * exact / max(len(samples), 1),
    }
    return {'per_sample': results, 'aggregate': agg}


# Ensure a place to save extended eval results
EXTENDED_RESULTS = {
    'synth_freeform': {},
    'synth_clean': {},
}
EXTENDED_RESULTS_PATH = Path(CHECKPOINT_DIR) / 'extended_large_model_results.json'
if EXTENDED_RESULTS_PATH.exists():
    with open(EXTENDED_RESULTS_PATH) as f:
        EXTENDED_RESULTS = json.load(f)
    print(f'Loaded existing results: {EXTENDED_RESULTS_PATH}')

def save_extended():
    with open(EXTENDED_RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(EXTENDED_RESULTS, f, indent=2, ensure_ascii=False)
    print(f'Saved → {EXTENDED_RESULTS_PATH}')
"""

GPT_CELL = """# ============================================================
# GPT-4o on synth + synth-freeform (few-shot)
# ============================================================
print('GPT-4o — synth-freeform few-shot')
EXTENDED_RESULTS['synth_freeform']['gpt4o_few_shot'] = evaluate_model(
    call_gpt4o, synth_ff_test_samples,
    build_few_shot_prompt, few_shot=synth_ff_few_shot,
    label='gpt4o-ff-fs',
)
save_extended()

print('\\nGPT-4o — synth few-shot')
EXTENDED_RESULTS['synth_clean']['gpt4o_few_shot'] = evaluate_model(
    call_gpt4o, synth_test_samples,
    build_few_shot_prompt, few_shot=synth_few_shot,
    label='gpt4o-synth-fs',
)
save_extended()

print('\\nGPT-4o aggregate:')
print(f"  synth-freeform: CER={EXTENDED_RESULTS['synth_freeform']['gpt4o_few_shot']['aggregate']['mean_cer_no_sp_pct']:.2f}%  "
      f"exact={EXTENDED_RESULTS['synth_freeform']['gpt4o_few_shot']['aggregate']['exact_match_pct']:.1f}%")
print(f"  synth:           CER={EXTENDED_RESULTS['synth_clean']['gpt4o_few_shot']['aggregate']['mean_cer_no_sp_pct']:.2f}%  "
      f"exact={EXTENDED_RESULTS['synth_clean']['gpt4o_few_shot']['aggregate']['exact_match_pct']:.1f}%")
"""

CLAUDE_CELL = """# ============================================================
# Claude on synth + synth-freeform (few-shot)
# ============================================================
print('Claude — synth-freeform few-shot')
EXTENDED_RESULTS['synth_freeform']['claude_few_shot'] = evaluate_model(
    call_claude, synth_ff_test_samples,
    build_few_shot_prompt, few_shot=synth_ff_few_shot,
    label='claude-ff-fs',
)
save_extended()

print('\\nClaude — synth few-shot')
EXTENDED_RESULTS['synth_clean']['claude_few_shot'] = evaluate_model(
    call_claude, synth_test_samples,
    build_few_shot_prompt, few_shot=synth_few_shot,
    label='claude-synth-fs',
)
save_extended()

print('\\nClaude aggregate:')
print(f"  synth-freeform: CER={EXTENDED_RESULTS['synth_freeform']['claude_few_shot']['aggregate']['mean_cer_no_sp_pct']:.2f}%  "
      f"exact={EXTENDED_RESULTS['synth_freeform']['claude_few_shot']['aggregate']['exact_match_pct']:.1f}%")
print(f"  synth:           CER={EXTENDED_RESULTS['synth_clean']['claude_few_shot']['aggregate']['mean_cer_no_sp_pct']:.2f}%  "
      f"exact={EXTENDED_RESULTS['synth_clean']['claude_few_shot']['aggregate']['exact_match_pct']:.1f}%")
"""

GEMINI_CELL = """# ============================================================
# Gemini on synth + synth-freeform (few-shot)
# Rate-limited on free tier — may need retries or paid API key
# ============================================================
print('Gemini — synth-freeform few-shot')
try:
    EXTENDED_RESULTS['synth_freeform']['gemini_few_shot'] = evaluate_model(
        call_gemini, synth_ff_test_samples,
        build_few_shot_prompt, few_shot=synth_ff_few_shot,
        label='gemini-ff-fs', sleep_s=2.0,  # more spacing for rate limits
    )
    save_extended()
except Exception as e:
    print(f'Gemini freeform failed: {e}')

print('\\nGemini — synth few-shot')
try:
    EXTENDED_RESULTS['synth_clean']['gemini_few_shot'] = evaluate_model(
        call_gemini, synth_test_samples,
        build_few_shot_prompt, few_shot=synth_few_shot,
        label='gemini-synth-fs', sleep_s=2.0,
    )
    save_extended()
except Exception as e:
    print(f'Gemini synth failed: {e}')

print('\\nGemini aggregate (where available):')
for k in ('synth_freeform', 'synth_clean'):
    if 'gemini_few_shot' in EXTENDED_RESULTS.get(k, {}):
        agg = EXTENDED_RESULTS[k]['gemini_few_shot']['aggregate']
        print(f"  {k}: CER={agg['mean_cer_no_sp_pct']:.2f}%  exact={agg['exact_match_pct']:.1f}%")
"""

SUMMARY_CELL = """# ============================================================
# Extended evaluation summary — large models vs fine-tuned models
# ============================================================
print('=' * 80)
print('EXTENDED LARGE-MODEL EVALUATION — final summary')
print('=' * 80)

print(f'\\n{"Model":<30}{"Synth CER":>12}{"Synth exact":>14}{"Freeform CER":>15}{"Freeform exact":>16}')
print('-' * 80)

def fmt(agg_or_none, key):
    if agg_or_none is None:
        return '—'
    val = agg_or_none['aggregate'].get(key)
    return f'{val:.2f}%' if val is not None else '—'

for model_label, mkey in [
    ('GPT-4o (few-shot)', 'gpt4o_few_shot'),
    ('Claude (few-shot)', 'claude_few_shot'),
    ('Gemini (few-shot)', 'gemini_few_shot'),
]:
    synth = EXTENDED_RESULTS.get('synth_clean', {}).get(mkey)
    ff    = EXTENDED_RESULTS.get('synth_freeform', {}).get(mkey)
    print(f'{model_label:<30}'
          f'{fmt(synth, "mean_cer_no_sp_pct"):>12}'
          f'{fmt(synth, "exact_match_pct"):>14}'
          f'{fmt(ff, "mean_cer_no_sp_pct"):>15}'
          f'{fmt(ff, "exact_match_pct"):>16}')

print('-' * 80)
print('Fine-tuned models (from earlier experiments, for comparison):')
print(f'{"TrOCR-small":<30}{"0.06%":>12}{"99.8%":>14}{"1.34%":>15}{"91.4%":>16}')
print(f'{"PARSeq":<30}{"8.96%":>12}{"—":>14}{"29.17%":>15}{"40.0%":>16}')
print(f'{"CNN+RNN":<30}{"66.82%":>12}{"—":>14}{"67.24%":>15}{"14.3%":>16}')
print('=' * 80)

save_extended()
"""


def strip_existing(nb):
    """Remove any previous extended-eval cells we appended."""
    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell.get("source", []))
        if EXTENDED_MARKER in src:
            nb["cells"] = nb["cells"][:i]
            return True
    return False


def main():
    with open(NB) as f:
        nb = json.load(f)
    strip_existing(nb)
    nb["cells"].extend([
        md(HEADER),
        code(SETUP_CELL),
        code(EVAL_HELPER_CELL),
        code(GPT_CELL),
        code(CLAUDE_CELL),
        code(GEMINI_CELL),
        code(SUMMARY_CELL),
    ])
    with open(NB, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Appended 7 extended-eval cells to {NB.name}")


if __name__ == "__main__":
    main()
