"""Audit dataset-statistics and Figure-4 claims from artifacts on disk.

Recomputes everything from evaluation_data/new_updated.xlsx, the 208 case
folders, and results/benchmark_std_post_fix/gemini-flash/<case>/metrics.json.
"""
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
XLSX = REPO / "evaluation_data/new_updated.xlsx"
RESULTS = REPO / "results/benchmark_std_post_fix/gemini-flash"
EVAL = REPO / "evaluation_data"

print("=" * 70)
print("SHEETS / SOURCES")
print("=" * 70)
df = pd.read_excel(XLSX, sheet_name="Cleaned_up_208_planning_dataset")
print("Cleaned_up_208_planning_dataset:", df.shape)

# ---------- (1) categorical counts ----------
print("\n" + "=" * 70)
print("(1) CATEGORICAL COUNTS (n=%d)" % len(df))
print("=" * 70)
for col in ["Shape Complexity", "Document Colour", "Document Quality",
            "Shape Matches correctly"]:
    vc = df[col].astype(str).str.strip().value_counts(dropna=False)
    print(f"\n{col}:")
    for k, v in vc.items():
        print(f"  {k!r}: {v}  ({v/len(df)*100:.1f}%)")

# case-insensitive normalised buckets
norm = lambda s: df[s].astype(str).str.strip().str.lower()
print("\nNormalised (lowercased):")
print(" complexity:", dict(norm("Shape Complexity").value_counts()))
print(" colour:    ", dict(norm("Document Colour").value_counts()))
print(" quality:   ", dict(norm("Document Quality").value_counts()))
print(" matches:   ", dict(norm("Shape Matches correctly").value_counts()))

# ---------- (2) curation arithmetic ----------
print("\n" + "=" * 70)
print("(2) CURATION ARITHMETIC")
print("=" * 70)
all270 = pd.read_excel(XLSX, sheet_name="All_270_planning_dataset_list")
mism = pd.read_excel(XLSX, sheet_name="Already Removed Shape Mismatch ")
other = pd.read_excel(XLSX, sheet_name="Other removals")
mrg = pd.read_excel(XLSX, sheet_name="Merged cases")
c230 = pd.read_excel(XLSX, sheet_name="230_cases_planning_dataset_list")
print("All_270 rows:", len(all270),
      "(non-null Unique ID:", all270["Unique ID (Folder_Name)"].notna().sum(), ")")
print("Shape Mismatch removals rows:", len(mism),
      "(non-null Unique ID:", mism["Unique ID (Folder_Name)"].notna().sum(), ")")
print("Other removals rows:", len(other))
print("Other removals reasons:")
print(other[["Unique ID (Folder_Name)", "Removal reason"]].to_string())
print("\nMerged cases sheet:")
print(mrg.to_string())
print("\n230-case sheet rows:", len(c230))

n_children = 0
for _, r in mrg.iterrows():
    kids = str(r["Children consolidated"])
    n_kid = len([k for k in re.split(r"[,;]", kids) if k.strip()])
    n_children += n_kid
print(f"\nMerged: {len(mrg)} merged folders, {n_children} children consolidated")
print("Check: 270-40-9-7 =", 270 - 40 - 9 - 7)
print("Check: 214-11+5 =", 214 - 11 + 5)

# ---------- (3) dates / geography ----------
print("\n" + "=" * 70)
print("(3) DATES & GEOGRAPHY")
print("=" * 70)
dates = pd.to_datetime(df["Document Date"], errors="coerce", dayfirst=True)
years = dates.dt.year
bad = df.loc[years.isna(), "Document Date"]
if len(bad):
    print("Unparsed dates:", bad.tolist())
    # try plain year extraction
    extra = df.loc[years.isna(), "Document Date"].astype(str).str.extract(r"(\d{4})")[0]
    years = years.fillna(pd.to_numeric(extra, errors="coerce"))
print("n with year:", years.notna().sum())
print("Year span:", int(years.min()), "-", int(years.max()))
print("Median year:", years.median())

def decade(y):
    if y < 1970: return "pre-1970"
    return f"{int(y // 10 * 10)}s"

dec = years.dropna().apply(decade).value_counts()
n = years.notna().sum()
for d in ["pre-1970", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s"]:
    c = dec.get(d, 0)
    print(f"  {d}: {c}  ({c/n*100:.1f}%)")

print("\nCounty column:")
county = df["County"].astype(str).str.strip()
vc = county.value_counts()
print("Distinct county values:", county.nunique())
for k, v in vc.items():
    print(f"  {k!r}: {v}  ({v/len(df)*100:.1f}%)")

# ---------- authorities from geojson organisation-entity ----------
print("\nLocal planning authorities (geojson 'organisation-entity'):")
case_dirs = sorted([p for p in EVAL.iterdir() if p.is_dir()])
print("case folders on disk:", len(case_dirs))
org_by_case = {}
no_geo = []
for d in case_dirs:
    gj = sorted(d.glob("*.geojson"))
    if not gj:
        no_geo.append(d.name)
        continue
    orgs = set()
    for g in gj:
        try:
            o = json.loads(g.read_text())
        except Exception as e:
            print("  parse fail", g, e)
            continue
        feats = o["features"] if o.get("type") == "FeatureCollection" else [o]
        for f in feats:
            p = f.get("properties", {}) or {}
            if "organisation-entity" in p:
                orgs.add(str(p["organisation-entity"]))
    org_by_case[d.name] = orgs
if no_geo:
    print("cases without geojson:", no_geo)
multi = {k: v for k, v in org_by_case.items() if len(v) > 1}
none = [k for k, v in org_by_case.items() if len(v) == 0]
if multi:
    print("cases with >1 org:", multi)
if none:
    print("cases with no org property:", none)
org_counter = Counter()
for k, v in org_by_case.items():
    if v:
        org_counter[sorted(v)[0]] += 1
print("Distinct organisation-entity values:", len(set().union(*[v for v in org_by_case.values() if v])))
print("Counts per org (sorted desc):")
for org, c in org_counter.most_common():
    print(f"  org {org}: {c}")

# ---------- (5) Figure 4 group values ----------
print("\n" + "=" * 70)
print("(5) FIGURE 4 RECOMPUTATION")
print("=" * 70)
# bridge merged folder names xlsx -> disk
bridge = dict(zip(mrg["Unnamed: 5"].astype(str), mrg["Merged folder"].astype(str)))
print("merged-name bridge:", bridge)
df["run_folder"] = df["Unique ID (Folder_Name)"].astype(str).map(lambda x: bridge.get(x, x))

iou_final, iou_first = {}, {}
missing_metrics = []
for f in df["run_folder"]:
    mp = RESULTS / f / "metrics.json"
    if not mp.exists():
        missing_metrics.append(f)
        continue
    m = json.loads(mp.read_text())
    iou_final[f] = m["iou"]
    iou_first[f] = m.get("worker_first_iou")
print("metrics.json found for", len(iou_final), "of", len(df), "xlsx cases")
if missing_metrics:
    print("MISSING metrics:", missing_metrics)

# also check result dirs not in xlsx
result_dirs = {p.name for p in RESULTS.iterdir() if p.is_dir()}
extra = result_dirs - set(df["run_folder"])
if extra:
    print("result dirs not matched by xlsx:", sorted(extra))
disk_cases = {p.name for p in case_dirs}
unmatched_disk = disk_cases - set(df["run_folder"])
if unmatched_disk:
    print("evaluation_data folders not matched by xlsx run_folder:", sorted(unmatched_disk))

df["iou"] = df["run_folder"].map(iou_final)
df["iou_wf"] = df["run_folder"].map(iou_first)
df["col_norm"] = norm("Document Colour")
df["colour_bucket"] = df["col_norm"].map(lambda x: x if x == "yellow" else "white")
df["quality_bucket"] = norm("Document Quality")
df["complexity_bucket"] = norm("Shape Complexity")

def table(col, order, ioucol):
    rows = [("Total", df)]
    rows += [(b, df[df[col] == b]) for b in order]
    out = []
    for lab, sub in rows:
        v = sub[ioucol].dropna()
        out.append((lab, len(sub), v.mean(), (v >= 0.8).mean()))
    return out

for ioucol, name in [("iou", "FINAL (with critic)"), ("iou_wf", "WORKER-FIRST (pre-critic)")]:
    print(f"\n--- {name} ---")
    for col, order, title in [
        ("colour_bucket", ["white", "yellow"], "Colour"),
        ("quality_bucket", ["good", "bad"], "Quality"),
        ("complexity_bucket", ["easy", "medium", "hard"], "Complexity"),
    ]:
        print(f" {title}:")
        for lab, n_, mean, frac in table(col, order, ioucol):
            print(f"   {lab:<8} n={n_:>3}  mean={mean:.4f} ({mean:.2f})  >=0.8: {frac*100:.1f}% ({frac*100:.0f}%)")

# compare with summary.json iou (what the figure script used)
summ = json.loads((RESULTS / "summary.json").read_text())
s_iou = {c["folder"]: c["iou"] for c in summ["per_case"]}
diff = [(f, s_iou.get(f), iou_final.get(f)) for f in df["run_folder"]
        if f in s_iou and abs((s_iou.get(f) or 0) - (iou_final.get(f) or 0)) > 1e-9]
print("\nsummary.json per_case iou vs metrics.json iou mismatches:", len(diff))
for t in diff[:10]:
    print("  ", t)
print("summary.json per_case count:", len(s_iou))
