import re, random
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def seed_everything(seed: int = 42):
    random.seed(seed); np.random.seed(seed)

def clean_text(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.replace("\u00A0"," ").replace("\u200b"," ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_bbc_dir(root: Path) -> pd.DataFrame:
    rows = []
    for label_dir in ["business","entertainment","politics","sport","tech"]:
        d = (root / label_dir)
        if not d.exists():
            print(f"[WARN] folder tidak ditemukan: {d}")
            continue
        for p in sorted(d.glob("*.txt")):
            txt = p.read_text(encoding="latin-1", errors="ignore")
            rows.append({"text": clean_text(txt), "label": label_dir, "path": str(p)})
    return pd.DataFrame(rows)

def load_summary_csv(csv_path: Path, text_col: str, summary_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    assert text_col in df.columns and summary_col in df.columns, \
        f"Kolom wajib tidak ditemukan. Ada: {list(df.columns)[:10]}"
    df = df[[text_col, summary_col]].rename(columns={text_col:"text", summary_col:"summary"})
    df["text"] = df["text"].map(clean_text)
    df["summary"] = df["summary"].map(clean_text)
    df = df.dropna().drop_duplicates()
    return df

def stratified_split(df: pd.DataFrame, label_col: str, test_size: float=0.2, seed: int=42):
    tr, te = train_test_split(df, test_size=test_size, random_state=seed, stratify=df[label_col])
    return tr.reset_index(drop=True), te.reset_index(drop=True)

def plot_confmat(y_true: List[str], y_pred: List[str], labels: List[str], title: str="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.colorbar(im)
    fig.tight_layout()
    return fig, ax
