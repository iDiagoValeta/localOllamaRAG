"""
inspect_splits.py -- Dataset split sizes for the 3 MonkeyGrab datasets.

Shows original HuggingFace split sizes and samples remaining after filters
F1 and F3. Aina is broken down by language (EN / ES / CA), yielding 5
datasets total.

Filters:
  F1  _filter_valid     -- instruction, context AND response must be non-empty.
  F3  _filter_dolly_rag -- Dolly only: category in {closed_qa,
                           information_extraction, summarization} AND
                           context non-empty.

Usage:
    python scripts/evaluation/inspect_splits.py

Dependencies:
    - datasets
"""

# ─────────────────────────────────────────────
# MODULE MAP -- Section index
# ─────────────────────────────────────────────
#
#  FILTERS
#  +-- 1. Normalizers and filter definitions
#
#  DATASET LOADING
#  +-- 2. Neural-Bridge RAG
#  +-- 3. Dolly 15k
#  +-- 4. Aina RAG Multilingual (per language)
#
#  REPORT
#  +-- 5. Summary table
#
# ─────────────────────────────────────────────

from datasets import load_dataset, get_dataset_split_names

_AINA_LANGS      = [("en", "Aina-EN"), ("es", "Aina-ES"), ("ca", "Aina-CA")]
_DOLLY_RAG_CATS  = {"closed_qa", "information_extraction", "summarization"}
_AINA_EXTRA_COLS = ["id", "category", "extractive"]


# ─────────────────────────────────────────────
# SECTION 1: NORMALIZERS AND FILTERS
# ─────────────────────────────────────────────

def _normalize_nb(ex):
    return {"instruction": (ex.get("question") or "").strip(),
            "context":     (ex.get("context")  or "").strip(),
            "response":    (ex.get("answer")   or "").strip()}

def _normalize_dolly(ex):
    return {"instruction": (ex.get("instruction") or "").strip(),
            "context":     (ex.get("context")     or "").strip(),
            "response":    (ex.get("response")    or "").strip(),
            "category":    (ex.get("category")    or "").strip()}

def _normalize_aina(ex):
    return {"instruction": (ex.get("instruction") or "").strip(),
            "context":     (ex.get("context")     or "").strip(),
            "response":    (ex.get("response")    or "").strip(),
            "lang":        (ex.get("lang")        or "").strip()}

def _filter_valid(ex):
    return bool(ex["instruction"]) and bool(ex["context"]) and bool(ex["response"])

def _filter_dolly_rag(ex):
    return ex["category"] in _DOLLY_RAG_CATS and bool(ex["context"])


# ─────────────────────────────────────────────
# SECTION 2: NEURAL-BRIDGE RAG
# ─────────────────────────────────────────────

print("Loading Neural-Bridge RAG ...")
nb = {}
for split in get_dataset_split_names("neural-bridge/rag-dataset-12000"):
    ds = load_dataset("neural-bridge/rag-dataset-12000", split=split)
    raw = len(ds)
    ds  = ds.map(_normalize_nb, remove_columns=["context", "question", "answer"]).filter(_filter_valid)
    nb[split] = (raw, len(ds))


# ─────────────────────────────────────────────
# SECTION 3: DOLLY 15K
# ─────────────────────────────────────────────

print("Loading Dolly 15k ...")
dolly = {}
for split in get_dataset_split_names("databricks/databricks-dolly-15k"):
    ds  = load_dataset("databricks/databricks-dolly-15k", split=split)
    raw = len(ds)
    ds  = (ds.map(_normalize_dolly, remove_columns=["instruction", "context", "response", "category"])
             .filter(_filter_dolly_rag)
             .filter(_filter_valid))
    dolly[split] = (raw, len(ds))


# ─────────────────────────────────────────────
# SECTION 4: AINA RAG MULTILINGUAL
# ─────────────────────────────────────────────

print("Loading Aina RAG Multilingual ...")
aina = {}   # split -> {lang_label: (raw_lang, filtered_lang)}
for split in get_dataset_split_names("projecte-aina/RAG_Multilingual"):
    ds_raw  = load_dataset("projecte-aina/RAG_Multilingual", split=split)
    raw_total = len(ds_raw)
    cols_drop = [c for c in _AINA_EXTRA_COLS if c in ds_raw.column_names]
    ds_f = ds_raw.map(_normalize_aina, remove_columns=cols_drop).filter(_filter_valid)
    aina[split] = {}
    for lang_code, lang_label in _AINA_LANGS:
        raw_lang = len(ds_raw.filter(lambda ex, lc=lang_code: ex["lang"] == lc))
        filt_lang = len(ds_f.filter(lambda ex, lc=lang_code: ex["lang"] == lc))
        aina[split][lang_label] = (raw_lang, filt_lang)


# ─────────────────────────────────────────────
# SECTION 5: SUMMARY TABLE
# ─────────────────────────────────────────────

W = 68
print()
print("=" * W)
print(f"  {'Dataset':<22} {'HF split':<14} {'HF original':>12} {'tras F1+F3':>12} {'retenido':>8}")
print(f"  {'-'*22} {'-'*14} {'-'*12} {'-'*12} {'-'*8}")

def _row(name, split, raw, filt):
    pct = f"{100*filt/raw:.1f} %" if raw else "-"
    print(f"  {name:<22} {split:<14} {raw:>12,} {filt:>12,} {pct:>8}")

# Neural-Bridge
for split, (raw, filt) in nb.items():
    _row("Neural-Bridge RAG", split, raw, filt)

# Dolly (only train)
for split, (raw, filt) in dolly.items():
    _row("Dolly QA", split, raw, filt)

# Aina per language
print(f"  {'-'*22} {'-'*14} {'-'*12} {'-'*12} {'-'*8}")
for split in ["train", "validation", "test"]:
    if split not in aina:
        continue
    for _, lang_label in _AINA_LANGS:
        raw, filt = aina[split][lang_label]
        _row(lang_label, split, raw, filt)
    print(f"  {'':<22} {'-'*14} {'-'*12} {'-'*12} {'-'*8}")

print("=" * W)
print()
print("  Filtros:")
print("    F1  campos vacios     -- instruction, context y response no vacios")
print("    F3  categoria Dolly   -- solo closed_qa / information_extraction / summarization")
print(f"                            con contexto no vacio (solo aplica a Dolly)")
print()
