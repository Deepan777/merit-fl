#!/usr/bin/env python3
"""Leakage audit: shallow decision trees on each
processed dataset, fit on train and scored on validation. The test split is
never read.

Per dataset:
  * all-feature tree (max_depth 4): accuracy, balanced accuracy, macro-F1,
    top feature importances;
  * single-feature trees (max_depth 3) for every kept feature and for every
    dropped identifier column (audit file);
  * flag "near-perfect": a single feature with macro-F1 >= 0.95 or accuracy
    >= 0.99;
  * investigation of every flagged or dominant kept feature (depth-4
    importance >= 0.5) and of every dropped identifier column: single-feature
    trees at depth 3 / 10 / unlimited, all-feature trees with and without the
    column, value purity (share of training rows whose exact value occurs in
    one class only; number of distinct values), per-class value bands.

python scripts/leakage_audit.py [--root data] [--taxonomy family]
Writes reports/leakage_audit.json and reports/LEAKAGE_AUDIT_TABLES.md.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.preprocess import SPECS, load_processed

MAX_TRAIN = 300_000
F1_FLAG, ACC_FLAG = 0.95, 0.99


def scores(y, p):
    return {"acc": accuracy_score(y, p), "bal_acc": balanced_accuracy_score(y, p),
            "macro_f1": f1_score(y, p, average="macro", zero_division=0)}


def tree(Xtr, ytr, Xva, yva, depth):
    t = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(Xtr, ytr)
    return t, scores(yva, t.predict(Xva))


def purity(x, y):
    """Share of rows whose exact value is seen in a single class; #values."""
    vals, inv = np.unique(x, return_inverse=True)
    n_cls = np.zeros(len(vals), int)
    for k in np.unique(y):
        n_cls += np.bincount(inv[y == k], minlength=len(vals)) > 0
    return {"single_class_value_share": float((n_cls[inv] == 1).mean()), "distinct_values": int(len(vals))}


def investigate(col_tr, col_va, ytr, yva, Xtr, Xva, kept_idx, classes):
    """Shortcut checks for one column (raw values). kept_idx: its index in X
    when it is a kept feature, else None (a dropped column is added)."""
    out = {"alone": {}, "without": {}, "with": {}}
    for d in (3, 10, None):
        out["alone"][str(d)] = tree(col_tr[:, None], ytr, col_va[:, None], yva, d)[1]
    if kept_idx is None:
        base_tr, base_va, plus_tr, plus_va = Xtr, Xva, np.c_[Xtr, col_tr], np.c_[Xva, col_va]
    else:
        rest = [j for j in range(Xtr.shape[1]) if j != kept_idx]
        base_tr, base_va, plus_tr, plus_va = Xtr[:, rest], Xva[:, rest], Xtr, Xva
    for d in (4, None):
        out["without"][str(d)] = tree(base_tr, ytr, base_va, yva, d)[1]
        out["with"][str(d)] = tree(plus_tr, ytr, plus_va, yva, d)[1]
    out.update(purity(col_tr, ytr))
    out["class_bands"] = {}
    for k, c in enumerate(classes):
        v = col_tr[ytr == k]
        if len(v):
            u, n = np.unique(v, return_counts=True)
            top = np.argsort(-n)[:3]
            out["class_bands"][c] = {"n": int(len(v)), "q01": float(np.quantile(v, .01)),
                                     "q50": float(np.quantile(v, .5)), "q99": float(np.quantile(v, .99)),
                                     "top3_values": [float(u[i]) for i in top],
                                     "top3_share": float(n[top].sum() / len(v))}
    return out


def audit(root, name, taxonomy):
    data, meta = load_processed(root, name, taxonomy, splits=("train", "val"))
    (Xtr, ytr), (Xva, yva) = data["train"], data["val"]
    if len(ytr) > MAX_TRAIN:
        keep, _ = train_test_split(np.arange(len(ytr)), train_size=MAX_TRAIN, stratify=ytr, random_state=0)
        Xtr, ytr = Xtr[keep], ytr[keep]
    else:
        keep = np.arange(len(ytr))
    feats = meta["features"]
    t, full = tree(Xtr, ytr, Xva, yva, 4)
    imp = sorted(zip(feats, t.feature_importances_), key=lambda z: -z[1])[:10]
    rows = []
    for j, f in enumerate(feats):
        _, s = tree(Xtr[:, [j]], ytr, Xva[:, [j]], yva, 3)
        rows.append({"feature": f, "kind": "kept", **s})
    apath = os.path.join(root, name, "processed", f"audit_{taxonomy}.npz")
    if os.path.exists(apath):
        with np.load(apath) as a:
            names = a["names"].tolist()
            Atr, Ava = a["X_train"][keep], a["X_val"]
        for j, f in enumerate(names):
            ok_tr, ok_va = np.isfinite(Atr[:, j]), np.isfinite(Ava[:, j])
            _, s = tree(Atr[ok_tr][:, [j]], ytr[ok_tr], Ava[ok_va][:, [j]], yva[ok_va], 3)
            rows.append({"feature": f, "kind": "dropped", "reason": meta["dropped_columns"][f], **s})
    rows.sort(key=lambda r: -r["macro_f1"])
    for r in rows:
        r["near_perfect"] = r["macro_f1"] >= F1_FLAG or r["acc"] >= ACC_FLAG
        if r["near_perfect"]:
            col = (Xtr[:, feats.index(r["feature"])] if r["kind"] == "kept"
                   else Atr[:, names.index(r["feature"])])
            r.update(purity(col, ytr))
    names = names if os.path.exists(apath) else []
    inv = {}
    dominant = {f for f, v in zip(feats, t.feature_importances_) if v >= 0.5}
    for r in rows:
        if r["kind"] == "kept" and (r["near_perfect"] or r["feature"] in dominant):
            j = feats.index(r["feature"])
            raw = lambda Z: Z[:, j].astype(np.float64) * meta["scaler"]["scale"][j] + meta["scaler"]["mean"][j]
            inv[r["feature"]] = {"kind": "kept (dominant or flagged)",
                                 **investigate(raw(Xtr), raw(Xva), ytr, yva, Xtr, Xva, j, meta["classes"])}
        elif r["kind"] == "dropped":
            j = names.index(r["feature"])
            ok_tr, ok_va = np.isfinite(Atr[:, j]), np.isfinite(Ava[:, j])
            if ok_tr.all() and ok_va.all():
                inv[r["feature"]] = {"kind": "dropped", **investigate(Atr[:, j], Ava[:, j], ytr, yva, Xtr, Xva,
                                                                      None, meta["classes"])}
    majority = float(np.bincount(yva).max() / len(yva))
    return {"dataset": name, "taxonomy": taxonomy, "classes": meta["classes"],
            "n_train_used": int(len(ytr)), "n_val": int(len(yva)),
            "majority_class_accuracy": majority, "all_features_depth4": full,
            "top_importances": [[f, float(v)] for f, v in imp], "single_feature_depth3": rows,
            "investigations": inv}


def to_markdown(results):
    f = lambda v: f"{v:.3f}"
    out = ["# Leakage audit tables (generated by scripts/leakage_audit.py)", "",
           f"Trees fit on train (≤ {MAX_TRAIN:,} rows, stratified), scored on validation; the test "
           f"split is not read. Flag: single feature with macro-F1 ≥ {F1_FLAG} or accuracy ≥ {ACC_FLAG}.", ""]
    for r in results:
        a = r["all_features_depth4"]
        out += [f"## {r['dataset']} ({r['taxonomy']}, {len(r['classes'])} classes)", "",
                f"Validation rows {r['n_val']:,}; majority-class accuracy {f(r['majority_class_accuracy'])}.", "",
                f"All features, depth 4: accuracy {f(a['acc'])}, balanced accuracy {f(a['bal_acc'])}, "
                f"macro-F1 {f(a['macro_f1'])}. Top importances: "
                + ", ".join(f"{n} ({v:.2f})" for n, v in r["top_importances"][:5]) + ".", "",
                "| Feature | Kept/dropped | Accuracy | Bal. acc. | Macro-F1 | Flag | Single-class value share | Distinct values |",
                "|---|---|---|---|---|---|---|---|"]
        for s in r["single_feature_depth3"][:15]:
            out.append(f"| {s['feature']} | {s['kind']} | {f(s['acc'])} | {f(s['bal_acc'])} | {f(s['macro_f1'])} | "
                       f"{'**near-perfect**' if s['near_perfect'] else ''} | "
                       f"{f(s['single_class_value_share']) if 'single_class_value_share' in s else ''} | "
                       f"{s.get('distinct_values', '')} |")
        flagged = [s["feature"] for s in r["single_feature_depth3"] if s["near_perfect"]]
        out += ["", f"Flagged: {', '.join(flagged) if flagged else 'none'}.", ""]
        for col, v in r["investigations"].items():
            out += [f"### Investigation: {col} ({v['kind']})", "",
                    "| Model (validation) | Accuracy | Bal. acc. | Macro-F1 |", "|---|---|---|---|"]
            for d, sc in v["alone"].items():
                out.append(f"| {col} alone, depth {d} | {f(sc['acc'])} | {f(sc['bal_acc'])} | {f(sc['macro_f1'])} |")
            for d in v["without"]:
                for tag in ("without", "with"):
                    sc = v[tag][d]
                    out.append(f"| all features {tag} {col}, depth {d} | {f(sc['acc'])} | {f(sc['bal_acc'])} | "
                               f"{f(sc['macro_f1'])} |")
            out += ["", f"Distinct values {v['distinct_values']:,}; training rows whose value occurs in one "
                        f"class only: {f(v['single_class_value_share'])}.", "",
                    "| Class | n (train) | q01 | median | q99 | top-3 values | top-3 share |", "|---|---|---|---|---|---|---|"]
            for c, b in v["class_bands"].items():
                out.append(f"| {c} | {b['n']:,} | {b['q01']:.6g} | {b['q50']:.6g} | {b['q99']:.6g} | "
                           f"{', '.join(f'{x:.6g}' for x in b['top3_values'])} | {f(b['top3_share'])} |")
            out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--taxonomy", default="family")
    ap.add_argument("--dataset", nargs="+", default=sorted(SPECS))
    ap.add_argument("--out", default="reports/leakage_audit", help="output prefix (.json and _TABLES.md)")
    a = ap.parse_args()
    t0 = time.time()
    res = []
    for name in a.dataset:
        if not os.path.exists(os.path.join(a.root, name, "processed", f"{a.taxonomy}.npz")):
            print(f"skip {name}: not preprocessed")
            continue
        res.append(audit(a.root, name, a.taxonomy))
        print(f"{name}: audited in {time.time() - t0:.0f} s")
    os.makedirs("reports", exist_ok=True)
    tables = "reports/LEAKAGE_AUDIT_TABLES.md" if a.out == "reports/leakage_audit" else a.out + "_TABLES.md"
    with open(a.out + ".json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    with open(tables, "w", encoding="utf-8") as fh:
        fh.write(to_markdown(res) + "\n")
    print(f"wrote {a.out}.json and {tables}")
