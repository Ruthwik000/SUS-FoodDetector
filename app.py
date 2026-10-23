"""
Streamlit front end for SusFoodDetector: upload a menu, pick a location, get adulteration risk per dish.

    streamlit run app.py

Only the menu (dish names + prices) and the location are asked for. Everything else the model was
trained on comes from:
  - the menu itself       -> ingredients (ingredient_nlp), price-vs-peer ratios, menu-level aggregates
  - the location          -> locations.csv and location_ingredient_stats.csv
  - a "typical" restaurant -> median / most common value of each restaurant attribute in training

Reference tables are read from $SUSFOOD_DATA_DIR (default: data/), the folder generate_dataset.py writes.
"""
import io
import os
import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from ingredient_nlp import IngredientInferer, expand_tracked_ingredients

DATA_DIR = Path(os.environ.get("SUSFOOD_DATA_DIR", "data"))
MODEL_PATH = Path(os.environ.get("SUSFOOD_MODEL", "sus_food_detector_lgbm.pkl"))

# Same set generate_dataset.py uses for share_high_risk_ingredient_items
HIGH_RISK = {"paneer", "khoya", "ghee", "soya", "milk", "saffron", "honey"}
LOCATION_COLS = ["city", "state", "city_tier", "area_type", "cost_of_living_index", "enforcement_intensity",
                 "avg_income_index", "population_density_k_per_km2", "overall_sample_fail_rate_12m"]
# Restaurant attributes we can't get from a menu; filled with training medians / modes
RESTAURANT_COLS = ["restaurant_type", "primary_cuisine", "fssai_licence", "hygiene_rating", "years_operating",
                   "seating_capacity", "online_rating", "num_reviews", "complaints_12m", "prior_violations_3y",
                   "months_since_last_inspection", "supplier_type", "on_delivery_apps", "avg_app_discount_pct"]

NAME_HINTS = ("dish", "item", "name", "menu", "food")
PRICE_HINTS = ("price", "cost", "rate", "inr", "amount", "rs", "mrp")
LINE_RE = re.compile(r"^\s*(?P<name>.*?[A-Za-z].*?)[\s.:|\-–—_]*(?:₹|rs\.?|inr)?\s*(?P<price>\d+(?:\.\d+)?)\s*(?:/-)?\s*$",
                     re.IGNORECASE)


# --------------------------------------------------------------------------------------
# Resources
# --------------------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model and reference data...")
def load_resources():
    bundle = joblib.load(MODEL_PATH)
    cats = bundle["categorical_features"]
    train = pd.read_csv(DATA_DIR / "training_items_flat.csv",
                        usecols=sorted(set(cats) | {"location_id", "menu_price_inr"}))
    categories = {c: sorted(train[c].dropna().unique()) for c in cats}
    peers = train.groupby(["city", "dish_name"])["menu_price_inr"].agg(["median", "size"])
    local = train.groupby(["location_id", "dish_name"])["menu_price_inr"].agg(["median", "size"])

    restaurants = pd.read_csv(DATA_DIR / "restaurants.csv")
    defaults = {}
    for c in RESTAURANT_COLS:
        col = restaurants[c]
        defaults[c] = col.mode().iloc[0] if c in cats else float(col.median())

    use_llm = bool(os.environ.get("ANTHROPIC_API_KEY"))
    return {
        "bundle": bundle, "categories": categories, "peers": peers, "local": local, "defaults": defaults,
        "locations": pd.read_csv(DATA_DIR / "locations.csv"),
        "loc_ing": pd.read_csv(DATA_DIR / "location_ingredient_stats.csv"),
        "nlp": IngredientInferer(use_llm=use_llm), "use_llm": use_llm,
    }


# --------------------------------------------------------------------------------------
# Menu parsing
# --------------------------------------------------------------------------------------

def _pick(columns, hints, exclude=None):
    for c in columns:
        if c != exclude and any(h in str(c).lower() for h in hints):
            return c
    return None


def parse_table(df):
    """Find the dish-name and price columns of an uploaded table."""
    df = df.dropna(how="all")
    name_col = _pick(df.columns, NAME_HINTS)
    price_col = _pick(df.columns, PRICE_HINTS, exclude=name_col)
    if name_col is None:
        name_col = next((c for c in df.columns if df[c].dtype == object), df.columns[0])
    if price_col is None:
        numeric = [c for c in df.columns if c != name_col and pd.to_numeric(df[c], errors="coerce").notna().any()]
        price_col = numeric[0] if numeric else None
    if price_col is None:
        raise ValueError("Couldn't find a price column. Name it 'price' (or 'cost' / 'rate').")
    prices = pd.to_numeric(df[price_col].astype(str).str.replace(r"[^\d.]", "", regex=True), errors="coerce")
    return pd.DataFrame({"dish_name": df[name_col].astype(str).str.strip(), "menu_price_inr": prices})


def parse_text(text):
    """One dish per line, price at the end: 'Paneer Tikka ..... ₹ 240'."""
    rows = []
    for line in text.splitlines():
        m = LINE_RE.match(line)
        if m:
            rows.append((m["name"].strip(" .:-|"), float(m["price"])))
        elif line.strip():
            rows.append((line.strip(), None))
    return pd.DataFrame(rows, columns=["dish_name", "menu_price_inr"])


def read_menu(upload):
    name = upload.name.lower()
    raw = upload.getvalue()
    if name.endswith((".xlsx", ".xls")):
        return parse_table(pd.read_excel(io.BytesIO(raw)))
    if name.endswith(".csv"):
        return parse_table(pd.read_csv(io.BytesIO(raw)))
    return parse_text(raw.decode("utf-8", errors="ignore"))


# --------------------------------------------------------------------------------------
# Scoring (mirrors "Scoring a new menu" in the notebook)
# --------------------------------------------------------------------------------------

def score_menu(menu, loc, res):
    bundle, categories = res["bundle"], res["categories"]
    menu = res["nlp"].enrich(menu.reset_index(drop=True))
    menu = menu.assign(location_id=loc["location_id"], **{c: loc[c] for c in LOCATION_COLS}, **res["defaults"])
    rows = expand_tracked_ingredients(menu, res["loc_ing"])
    if rows.empty:
        return menu, rows, None

    # Peer prices exist only for dishes the training data has seen (by canonical name)
    rows["dish_name"] = rows["nlp_matched_dish"].where(rows["nlp_matched_dish"].isin(categories["dish_name"]))
    for prefix, table, key in [("peer", res["peers"], "city"), ("local", res["local"], "location_id")]:
        idx = pd.MultiIndex.from_frame(rows[[key, "dish_name"]])
        median_col = "peer_median_price_inr" if prefix == "peer" else "local_median_price_inr"
        count_col = "peer_group_size" if prefix == "peer" else "local_listing_count"
        rows[median_col] = table["median"].reindex(idx).values
        rows[count_col] = table["size"].reindex(idx).values
    rows["price_ratio_to_peer_median"] = rows["menu_price_inr"] / rows["peer_median_price_inr"]
    rows["price_ratio_to_local_median"] = rows["menu_price_inr"] / rows["local_median_price_inr"]
    rows["quality_complaint_mentions"] = 0
    # The model never saw a missing peer price and reads one as "very cheap" (Butter Roti 3% -> 75%),
    # so a dish with no peers is treated as priced at market; the shown ratio stays blank.
    rows["shown_ratio"] = rows["price_ratio_to_peer_median"]
    for median_col, count_col, ratio_col in [
            ("peer_median_price_inr", "peer_group_size", "price_ratio_to_peer_median"),
            ("local_median_price_inr", "local_listing_count", "price_ratio_to_local_median")]:
        rows[median_col] = rows[median_col].fillna(rows["menu_price_inr"])
        rows[count_col] = rows[count_col].fillna(0)
        rows[ratio_col] = rows[ratio_col].fillna(1.0)

    # Menu-level aggregates the model sees as restaurant features (computed on each dish's primary ingredient)
    primary = rows[rows["component"] == 0]
    rows["menu_size"] = len(menu)
    rows["avg_menu_price_inr"] = menu["menu_price_inr"].mean()
    rows["share_high_risk_ingredient_items"] = menu["primary_ingredient"].isin(HIGH_RISK).sum() / len(menu)
    rows["share_items_on_local_hotspot_ingredient"] = primary["loc_ing_is_known_hotspot"].fillna(0).sum() / len(menu)
    rows["mean_price_ratio_to_peer"] = primary["price_ratio_to_peer_median"].mean()

    X = rows.reindex(columns=bundle["features"])
    for c in bundle["categorical_features"]:
        X[c] = pd.Categorical(X[c], categories=categories[c])
    rows["p"] = bundle["model"].predict_proba(X)[:, 1]

    # A dish is clean only if every risky ingredient in it is: P(any) = 1 - prod(1 - p_i)
    per_item = rows.groupby("item_index").apply(lambda g: pd.Series({
        "risk": 1 - (1 - g["p"]).prod(),
        "riskiest_ingredient": g.loc[g["p"].idxmax(), "primary_ingredient"],
        "ingredient_scores": ", ".join(f"{i} {p:.0%}" for i, p in zip(g["primary_ingredient"], g["p"])),
        "price_vs_peers": g["shown_ratio"].iloc[0],
    }), include_groups=False)
    report = menu.join(per_item)
    report["flagged"] = report["risk"] >= bundle["threshold"]
    return menu, rows, report


# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------

st.set_page_config(page_title="SusFoodDetector", page_icon="🤨", layout="wide")
st.title("🤨 SusFoodDetector")
st.caption("Upload a restaurant menu and pick its location to see which dishes are most likely to be adulterated.")

missing = [p for p in (MODEL_PATH, DATA_DIR / "training_items_flat.csv") if not p.exists()]
if missing:
    st.error(f"Missing {', '.join(map(str, missing))}. Point SUSFOOD_DATA_DIR at the folder generate_dataset.py "
             "wrote (it needs training_items_flat.csv, restaurants.csv, locations.csv and "
             "location_ingredient_stats.csv).")
    st.stop()
res = load_resources()
locations = res["locations"]

col_menu, col_loc = st.columns([3, 2], gap="large")

with col_loc:
    st.subheader("📍 Location")
    city = st.selectbox("City", sorted(locations["city"].unique()))
    city_locs = locations[locations["city"] == city]
    area = st.selectbox("Area", city_locs["area_type"].tolist(),
                        format_func=lambda a: a.replace("_", " ").capitalize())
    loc = city_locs[city_locs["area_type"] == area].iloc[0]
    hotspots = str(loc["known_hotspot_ingredients"]) if pd.notna(loc["known_hotspot_ingredients"]) else ""
    a, b = st.columns(2)
    a.metric("Lab samples failed (12m)", f"{loc['overall_sample_fail_rate_12m']:.0%}")
    b.metric("Seizures (12m)", int(loc["total_seizures_12m"]))
    st.write("**Known hotspot ingredients:** " + (", ".join(h.replace("_", " ") for h in hotspots.split(";"))
                                                 if hotspots else "none reported"))

with col_menu:
    st.subheader("📄 Menu")
    upload = st.file_uploader("Upload a menu", type=["csv", "xlsx", "xls", "txt"],
                              help="CSV / Excel with a dish-name and a price column, or a text file with one "
                                   "dish per line and the price at the end (e.g. `Paneer Tikka - 240`).")
    if upload is None:
        st.info("No menu yet. CSV/Excel needs a name and a price column; a .txt needs one `Dish name  price` "
                "per line.")
        st.stop()
    try:
        menu_in = read_menu(upload)
    except Exception as e:
        st.error(f"Couldn't read {upload.name}: {e}")
        st.stop()

    no_price = menu_in["menu_price_inr"].isna()
    if no_price.any():
        st.warning(f"{no_price.sum()} line(s) had no price and are skipped: "
                   + ", ".join(menu_in.loc[no_price, "dish_name"].head(8)))
    menu_in = menu_in[~no_price & (menu_in["dish_name"] != "")]
    with st.expander(f"Parsed {len(menu_in)} dishes (you can fix names or prices here)"):
        menu_in = st.data_editor(menu_in.reset_index(drop=True), num_rows="dynamic", width="stretch",
                                 column_config={"dish_name": "Dish",
                                                "menu_price_inr": st.column_config.NumberColumn("Price (₹)")})
    menu_in = menu_in.dropna()

if menu_in.empty:
    st.warning("No dishes with prices found in this menu.")
    st.stop()

with st.spinner("Scoring menu..."):
    menu, rows, report = score_menu(menu_in, loc, res)

st.divider()
if report is None:
    st.warning("None of these dishes contain an ingredient the model tracks (paneer, ghee, milk, oil, spices ...).")
    st.stop()

scored = report[report["risk"].notna()]
flagged = scored[scored["flagged"]]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Restaurant risk", f"{scored['risk'].mean():.0%}", help="Mean dish risk, as the model was tuned on")
m2.metric("Dishes flagged", f"{len(flagged)} / {len(scored)}")
m3.metric("Highest-risk dish", f"{scored['risk'].max():.0%}")
m4.metric("Not scored", len(report) - len(scored), help="No tracked risky ingredient, or the name wasn't recognised")

st.subheader("Dish risk")
view = scored.sort_values("risk", ascending=False)[[
    "dish_name", "menu_price_inr", "risk", "flagged", "riskiest_ingredient", "ingredient_scores",
    "price_vs_peers", "nlp_matched_dish", "veg_nonveg"]]
st.dataframe(view, hide_index=True, width="stretch", column_config={
    "dish_name": "Dish",
    "menu_price_inr": st.column_config.NumberColumn("Price", format="₹%d"),
    "risk": st.column_config.ProgressColumn("Adulteration risk", format="percent", min_value=0, max_value=1),
    "flagged": st.column_config.CheckboxColumn("Flagged"),
    "riskiest_ingredient": "Riskiest ingredient",
    "ingredient_scores": "Per-ingredient risk",
    "price_vs_peers": st.column_config.NumberColumn("Price vs. local peers", format="%.2f×",
                                                    help="Below 1 = cheaper than the same dish in this city"),
    "nlp_matched_dish": "Recognised as",
    "veg_nonveg": "Veg / non-veg",
})

unscored = report[report["risk"].isna()]
if not unscored.empty:
    with st.expander(f"{len(unscored)} dishes not scored"):
        st.dataframe(unscored[["dish_name", "menu_price_inr", "nlp_matched_dish", "nlp_ingredients", "nlp_source"]],
                     hide_index=True, width="stretch",
                     column_config={"nlp_ingredients": "Ingredients found", "nlp_source": "Matched by"})

st.download_button("Download results (CSV)", report.drop(columns=["location_id"]).to_csv(index=False),
                   file_name=f"sus_food_report_{city}_{area}.csv", mime="text/csv")

with st.expander("How this is scored / assumptions"):
    d = res["defaults"]
    st.markdown(
        f"- Ingredients are inferred from each dish name (`ingredient_nlp.py`; Claude fallback "
        f"{'**on**' if res['use_llm'] else 'off, set `ANTHROPIC_API_KEY` to enable'}). A dish is scored once per "
        "risky ingredient and combined as P(any) = 1 − ∏(1 − pᵢ).\n"
        f"- A dish is flagged at risk ≥ {res['bundle']['threshold']:.0%} (threshold tuned for F1 on validation).\n"
        "- Prices are compared with the same dish in the chosen city and area from the training data; dishes the "
        "training data never saw have no peer comparison and are treated as priced at market.\n"
        "- Since only the menu and location are given, the restaurant is assumed to be a typical one: "
        + ", ".join(f"{k.replace('_', ' ')} = {v:g}" if isinstance(v, float) else f"{k.replace('_', ' ')} = {v}"
                    for k, v in d.items()) + ".\n"
        "- The model was trained on synthetic data, so treat scores as a prioritisation aid, not a lab result.")
