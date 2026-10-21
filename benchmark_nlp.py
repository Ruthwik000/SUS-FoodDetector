#!/usr/bin/env python3
"""
Benchmark for ingredient_nlp.py.

  1. Robustness  - training dish names corrupted the way real menus are (typos, spelling variants,
                   filler like "(Full)" / "Chef's Special"), resolved by the NLP layer vs. naive baselines.
  2. Hidden ingredients - hand-labelled dishes whose names don't say what's in them.
  3. Downstream  - test-set ROC-AUC / PR-AUC when the model gets its ingredient columns from messy
                   names through each resolver.
  4. Throughput.

Usage:
    python3 benchmark_nlp.py --data data --model sus_food_detector_lgbm.pkl
"""
import argparse
import random
import re
import time

import joblib
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from sklearn.metrics import average_precision_score, roc_auc_score

from ingredient_nlp import MEAT, TRACKED_INGREDIENTS, IngredientInferer

ING_COLS = ["primary_ingredient", "ingredient_grams_est", "veg_nonveg", "dish_category"]

# Common Indian-English spellings of words in the training dish names
SPELLINGS = {
    "paneer": ["panir", "paner"], "dal": ["daal", "dhal"], "chicken": ["murgh", "chiken"],
    "biryani": ["biriyani", "briyani"], "makhani": ["makhni"], "matar": ["mutter", "mattar"],
    "gobi": ["gobhi"], "aloo": ["alu"], "fish": ["machli"], "chai": ["chaay"], "pakora": ["pakoda"],
    "tikka": ["tika"], "jamun": ["jamoon"], "ladoo": ["laddu"], "chole": ["chhole"], "puri": ["poori"],
    "vada": ["wada"], "rabri": ["rabdi"], "curd": ["dahi"], "masala": ["masaala"], "kadai": ["kadhai"],
    "bhature": ["bhatoore"], "tea": ["chai"], "chilli": ["chilly", "chili"], "dahi": ["curd"],
    "kheer": ["khir"], "khichdi": ["khichri"], "sambar": ["sambhar"], "halwa": ["halva"],
}
PREFIXES = ["Special ", "Chef's ", "Spl. ", "House ", "Homestyle "]
SUFFIXES = [" (Full)", " (Half)", " - 2 pcs", " [Bestseller]", " 500ml", " Plate", " (Serves 2)"]

# Hand-labelled: (menu text, veg_nonveg, risky tracked ingredients a food-safety officer would test)
HIDDEN = [
    ("Hyderabadi Haleem", "nonveg", {"ghee"}),
    ("Mutton Nihari", "nonveg", {"ghee"}),
    ("Dal Bukhara", "veg", {"butter"}),
    ("Murgh Makhanwala", "nonveg", {"butter", "chicken"}),
    ("Rasgulla", "veg", {"paneer"}),
    ("Sandesh", "veg", {"paneer"}),
    ("Chhena Poda", "veg", {"paneer"}),
    ("Shahi Tukda", "veg", {"milk", "ghee", "saffron"}),
    ("Ghevar", "veg", {"ghee", "milk"}),
    ("Rogan Josh", "nonveg", {"curd", "ghee", "red_chilli_powder"}),
    ("Galouti Kebab", "nonveg", {"ghee"}),
    ("Mutton Galouti", "nonveg", {"ghee"}),
    ("Kadhi Chawal", "veg", {"curd"}),
    ("Mishti Doi", "veg", {"curd", "milk"}),
    ("Navratan Korma", "veg", {"paneer"}),
    ("Sarson da Saag", "veg", {"butter", "ghee"}),
    ("Pindi Chole", "veg", {"edible_oil"}),
    ("Kaju Katli", "veg", {"ghee"}),
    ("Shrikhand", "veg", {"curd", "saffron"}),
    ("Thandai", "veg", {"milk", "saffron"}),
    ("Kulfi Falooda", "veg", {"milk"}),
    ("Malai Kofta", "veg", {"khoya", "paneer", "edible_oil"}),
    ("Gajar ka Halwa", "veg", {"khoya", "ghee", "milk"}),
    ("Butter Garlic Prawns", "nonveg", {"butter", "fish"}),
    ("Patra ni Machhi", "nonveg", {"fish"}),
    ("Chicken Ghee Roast", "nonveg", {"chicken", "ghee"}),
    ("Chicken Reshmi Kebab", "nonveg", {"chicken", "curd"}),
    ("Chicken Chettinad", "nonveg", {"chicken"}),
    ("Chicken Kolhapuri", "nonveg", {"chicken", "red_chilli_powder"}),
    ("Paneer Pasanda", "veg", {"paneer"}),
    ("Paneer 65", "veg", {"paneer", "red_chilli_powder", "edible_oil"}),
    ("Khoya Paneer", "veg", {"khoya", "paneer"}),
    ("Ghee Podi Idli", "veg", {"ghee"}),
    ("Kesar Jalebi", "veg", {"saffron", "edible_oil"}),
    ("Rabri Jalebi", "veg", {"milk", "edible_oil"}),
    ("Amul Butter Pav Bhaji", "veg", {"butter"}),
    ("Veg Kolhapuri", "veg", {"red_chilli_powder"}),
    ("Undhiyu", "veg", {"edible_oil"}),
    ("Kosha Mangsho", "nonveg", {"edible_oil"}),
    ("Bhapa Ilish", "nonveg", {"fish"}),
    ("Litti Chokha", "veg", {"ghee"}),
]


# --------------------------------------------------------------------------------------
# Perturbations
# --------------------------------------------------------------------------------------

def typo(rng, name):
    words = name.split()
    idx = [i for i, w in enumerate(words) if len(w) >= 5 and w.isalpha()]
    if not idx:
        return name
    i = rng.choice(idx)
    w = words[i]
    j = rng.randrange(1, len(w) - 1)
    op = rng.choice(["delete", "swap", "double", "vowel"])
    if op == "delete":
        w = w[:j] + w[j + 1:]
    elif op == "swap":
        w = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    elif op == "double":
        w = w[:j] + w[j] + w[j:]
    else:
        vowels = [k for k, c in enumerate(w) if c in "aeiou" and k > 0]
        if vowels:
            k = rng.choice(vowels)
            w = w[:k] + rng.choice([v for v in "aeiou" if v != w[k]]) + w[k + 1:]
    words[i] = w
    return " ".join(words)


def respell(rng, name):
    words = name.split()
    cand = [i for i, w in enumerate(words) if w.lower() in SPELLINGS]
    if not cand:
        return name
    i = rng.choice(cand)
    words[i] = rng.choice(SPELLINGS[words[i].lower()]).capitalize()
    return " ".join(words)


def filler(rng, name):
    return rng.choice(PREFIXES + [""]) + name + rng.choice(SUFFIXES)


def case(rng, name):
    return rng.choice([name.lower(), name.upper()])


def mixed(rng, name):
    for f in rng.sample([typo, respell, filler, case], k=rng.randint(2, 3)):
        name = f(rng, name)
    return name


PERTURBATIONS = {"clean": lambda rng, n: n, "case": case, "filler": filler, "spelling": respell,
                 "typo": typo, "mixed (2-3 of the above)": mixed}


# --------------------------------------------------------------------------------------
# Resolvers: menu text -> (canonical dish or None, {four ingredient columns})
# --------------------------------------------------------------------------------------

class Baselines:
    def __init__(self, dish_table):
        self.table = dish_table.set_index("dish_name")[ING_COLS]
        self.norm = {self._n(d): d for d in self.table.index}
        self.lower = {d.lower(): d for d in self.table.index}

    @staticmethod
    def _n(s):
        return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", s.lower())).strip()

    def _row(self, dish):
        return (dish, self.table.loc[dish].to_dict()) if dish else (None, dict.fromkeys(ING_COLS))

    def exact(self, name):
        return self._row(name if name in self.table.index else None)

    def normalised(self, name):
        return self._row(self.norm.get(self._n(name)))

    def fuzzy(self, name):
        hit = process.extractOne(name.lower(), list(self.lower), scorer=fuzz.WRatio, score_cutoff=85)
        return self._row(self.lower[hit[0]] if hit else None)


def nlp_resolver(nlp, known_dishes):
    def resolve(name):
        p = nlp.infer(name)
        dish = p.matched_dishes[0] if len(p.matched_dishes) == 1 and p.matched_dishes[0] in known_dishes else None
        return dish, {"primary_ingredient": p.primary_ingredient, "ingredient_grams_est": p.primary_grams,
                      "veg_nonveg": p.veg_nonveg, "dish_category": p.dish_category}
    return resolve


def keyword_only(nlp, name):
    strong, weak = nlp._keywords(nlp._clean(name))
    ings = [i for i in list(strong) + list(weak) if i]
    return {"tracked": {i for i in ings if i in TRACKED_INGREDIENTS},
            "veg_nonveg": ("nonveg" if MEAT & set(ings) else "veg") if ings else None}


# --------------------------------------------------------------------------------------

def table(df, title):
    print(f"\n### {title}\n")
    print(df.to_markdown(floatfmt=".3f") if hasattr(df, "to_markdown") else df.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--model", default="sus_food_detector_lgbm.pkl")
    ap.add_argument("--variants", type=int, default=10, help="noisy variants per dish per perturbation")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(f"{args.data}/training_items_flat.csv")
    dish_table = df[["dish_name"] + ING_COLS].drop_duplicates("dish_name").reset_index(drop=True)
    known = set(dish_table["dish_name"])
    nlp = IngredientInferer(use_llm=False, cache_path=None)
    base = Baselines(dish_table)
    resolvers = {"exact lookup": base.exact, "normalised lookup": base.normalised,
                 "fuzzy match (WRatio>=85)": base.fuzzy, "NLP layer": nlp_resolver(nlp, known)}

    # ---- 1. robustness ----
    rng = random.Random(args.seed)
    gold = dish_table.set_index("dish_name")[ING_COLS]
    cases = [(kind, dish, f(rng, dish)) for kind, f in PERTURBATIONS.items()
             for dish in dish_table["dish_name"] for _ in range(1 if kind == "clean" else args.variants)]
    rows = []
    for rname, resolve in resolvers.items():
        for kind, dish, text in cases:
            _, cols = resolve(text)
            g = gold.loc[dish]
            rows.append({"resolver": rname, "perturbation": kind,
                         "all 4 columns": all(cols[c] == g[c] for c in ING_COLS),
                         "primary_ingredient": cols["primary_ingredient"] == g["primary_ingredient"],
                         "veg_nonveg": cols["veg_nonveg"] == g["veg_nonveg"]})
    rob = pd.DataFrame(rows)
    print(f"# NLP benchmark\n\n{len(dish_table)} training dishes, {len(cases)} test strings "
          f"({args.variants} variants per dish per perturbation, seed {args.seed})")
    table(rob.pivot_table(index="perturbation", columns="resolver", values="all 4 columns",
                          sort=False)[list(resolvers)],
          "1. Robustness: share of menu strings where all four ingredient columns are correct")
    table(rob.groupby("resolver", sort=False)[["all 4 columns", "primary_ingredient", "veg_nonveg"]].mean(),
          "1b. Robustness, all perturbations pooled")
    fails = [(t, d) for (k, d, t), ok in zip(cases, rob[rob.resolver == "NLP layer"]["all 4 columns"]) if not ok]
    print(f"\nNLP misses: {len(fails)}. Examples: " + "; ".join(f"{t!r} (was {d})" for t, d in fails[:8]))

    # ---- 2. hidden ingredients ----
    hrows = []
    for name, veg, risky in HIDDEN:
        p = nlp.infer(name)
        kw = keyword_only(nlp, name)
        _, ex = base.fuzzy(name)
        fz_dish = base.fuzzy(name)[0]
        fz_tracked = {ex["primary_ingredient"]} if fz_dish else set()
        for rname, tracked, v, src in [("NLP layer", set(p.tracked_ingredients), p.veg_nonveg, p.source),
                                       ("keywords only", kw["tracked"], kw["veg_nonveg"], ""),
                                       ("fuzzy match to training dishes", fz_tracked, ex["veg_nonveg"], "")]:
            hrows.append({"resolver": rname, "dish": name, "source": src, "resolved": bool(tracked or v),
                          "veg_ok": v == veg, "tp": len(tracked & risky), "n_pred": len(tracked),
                          "n_gold": len(risky), "all_found": risky <= tracked})
    hid = pd.DataFrame(hrows)
    summ = hid.groupby("resolver", sort=False).apply(lambda g: pd.Series({
        "resolved": g["resolved"].mean(), "veg/nonveg acc": g["veg_ok"].mean(),
        "risky-ingredient recall": g["tp"].sum() / g["n_gold"].sum(),
        "risky-ingredient precision": g["tp"].sum() / max(g["n_pred"].sum(), 1),
        "all risky found": g["all_found"].mean()}), include_groups=False)
    table(summ, f"2. Hidden ingredients ({len(HIDDEN)} hand-labelled dishes)")
    nl = hid[hid.resolver == "NLP layer"]
    table(nl.groupby("source")[["veg_ok", "all_found"]].agg(["mean", "size"]).round(3),
          "2b. NLP layer by resolution tier")
    misses = nl[~nl["all_found"]]
    print("\nNLP layer missed some risky ingredient for: " + ", ".join(misses["dish"]))

    # ---- 3. downstream ----
    bundle = joblib.load(args.model)
    model, feats, cats = bundle["model"], bundle["features"], bundle["categorical_features"]
    X = df[feats].copy()
    X[cats] = X[cats].astype("category")
    te = (df["split"] == "test").values
    y = df.loc[te, "is_adulterated"].values
    rng = random.Random(args.seed + 1)
    noisy = [mixed(rng, d) for d in df.loc[te, "dish_name"]]

    def score(Xt):
        p = model.predict_proba(Xt)[:, 1]
        return roc_auc_score(y, p), average_precision_score(y, p), (p >= bundle["threshold"]).mean()

    # Everything derived from the ingredient is rebuilt from what the resolver says, not the truth:
    # the four columns, dish_name, the location x ingredient stats and the ingredient-cost features.
    li = pd.read_csv(f"{args.data}/location_ingredient_stats.csv")
    li = li.rename(columns={c: f"loc_ing_{c}" for c in li.columns if c not in ("location_id", "ingredient")})
    li_cols = [c for c in li.columns if c.startswith("loc_ing_") and c in feats]
    test_rows = df.loc[te, ["location_id", "menu_price_inr"]].reset_index(drop=True)

    def build(dish, cols):
        Xt = X[te].copy().reset_index(drop=True)
        for c in ING_COLS:
            Xt[c] = pd.Categorical(cols[c], categories=X[c].cat.categories) if c in cats else pd.to_numeric(cols[c])
        Xt["dish_name"] = pd.Categorical(dish, categories=X["dish_name"].cat.categories)
        stats = test_rows.assign(ingredient=list(cols["primary_ingredient"])).merge(
            li, on=["location_id", "ingredient"], how="left")
        Xt[li_cols] = stats[li_cols].values
        cost = Xt["ingredient_grams_est"] / 1000 * Xt["loc_ing_market_price_per_kg"]
        Xt["est_ingredient_cost_inr"] = cost.round(2)
        Xt["price_to_ingredient_cost_ratio"] = (test_rows["menu_price_inr"] / cost.round(2).clip(lower=0.5)).round(3)
        return Xt

    truth = {c: df.loc[te, c].tolist() for c in ING_COLS}
    rebuilt = build(df.loc[te, "dish_name"].tolist(), truth)
    gap = np.abs(model.predict_proba(rebuilt)[:, 1] - model.predict_proba(X[te])[:, 1])
    print(f"\nRebuild check (true ingredient -> rebuilt features): max |dp| {gap.max():.4f}, "
          f"{(gap > 1e-6).mean():.2%} of items differ (cent-level rounding of ingredient cost)")
    drows = [("original columns (oracle)", *score(X[te]), 1.0)]
    drows.append(("no ingredient info at all",
                  *score(build([None] * te.sum(), {c: [None] * te.sum() for c in ING_COLS})), 0.0))
    for rname, resolve in resolvers.items():
        memo = {t: resolve(t) for t in set(noisy)}
        cols = {c: [memo[t][1][c] for t in noisy] for c in ING_COLS}
        recovered = np.mean([a == g for a, g in zip(cols["primary_ingredient"], truth["primary_ingredient"])])
        drows.append((f"messy names -> {rname}", *score(build([memo[t][0] for t in noisy], cols)), recovered))
    table(pd.DataFrame(drows, columns=["ingredient columns from", "ROC-AUC", "PR-AUC", "flag rate",
                                       "primary_ingredient correct"]).set_index("ingredient columns from"),
          f"3. Downstream: {te.sum()} test items, dish names corrupted with 2-3 perturbations each")

    # ---- 4. throughput ----
    names = [mixed(random.Random(i), d) for i, d in enumerate(dish_table["dish_name"].tolist() * 20)]
    fresh = IngredientInferer(cache_path=None)
    t0 = time.perf_counter()
    for n in names:
        fresh.infer(n)
    per = (time.perf_counter() - t0) / len(names)
    t0 = time.perf_counter()
    fresh.enrich(df[["dish_name"]])
    full = time.perf_counter() - t0
    print(f"\n### 4. Throughput\n\n- {1 / per:,.0f} unique messy names/sec ({per * 1e3:.2f} ms each, "
          f"{len(names)} names)\n- enrich() on all {len(df):,} dataset rows: {full:.2f}s "
          f"(names are resolved once per unique string)")


if __name__ == "__main__":
    main()
