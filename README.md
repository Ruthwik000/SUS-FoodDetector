# SusFoodDetector 🤨

Flags restaurant dishes that are likely to be adulterated (watered-down milk, starch in paneer, dyed chilli
powder ...). Give it a menu and a location; it tells you which dishes look suspicious and why.

> Trained on a **synthetic** dataset of 8,000 restaurants / 127K menu items. Treat scores as a way to
> prioritise inspections, not as a lab result.

## How it works

1. **Ingredients from dish names** (`ingredient_nlp.py`): a menu only lists names, and many hide what's
   inside (Haleem → mutton + ghee, Rasgulla → paneer). A knowledge base of ~200 Indian dishes (340+ names
   and aliases), spelling normalisation and fuzzy matching resolve each name to its risky ingredients.
   Claude can be used as a fallback for names nothing else recognises.
2. **Risk model** (`sus_food_detector_lgbm.pkl`): a LightGBM classifier over price-vs-peer signals, location ×
   ingredient lab-test and market stats, and restaurant attributes. Each dish is scored once per risky
   ingredient and combined as P(any) = 1 − ∏(1 − pᵢ).
3. **App** (`app.py`): Streamlit UI that takes just a menu and a location.

| | ROC-AUC | PR-AUC |
|---|---|---|
| LightGBM, test set | 0.902 | 0.762 |
| Same, with ingredients inferred from messy dish names | 0.902 | 0.760 |

The NLP layer recovers all ingredient columns for 98.7% of misspelled / noisy menu strings (vs. 12.5% for exact
lookup). Full numbers in [`benchmark_results.md`](benchmark_results.md).

## Run the app

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

The app needs the dataset tables in `data/` (or set `SUSFOOD_DATA_DIR` to another folder):
`training_items_flat.csv`, `restaurants.csv`, `locations.csv`, `location_ingredient_stats.csv`.
They are produced by `generate_dataset.py --restaurants 8000 --seed 42 --out data`.

Set `ANTHROPIC_API_KEY` to turn on the Claude fallback for unrecognised dish names.

### Using it

- **Menu:** CSV or Excel with a dish-name column and a price column, or a `.txt` with one dish per line and the
  price at the end (`Paneer Butter Masala ..... ₹ 120`). Parsed rows can be corrected in the app.
- **Location:** city and area type (railway station, old city market, IT corridor ...).
- **Output:** restaurant risk, per-dish risk and flag, riskiest ingredient, price vs. the same dish in the
  city, and a CSV download.

Restaurant details the model was trained on but a menu doesn't give (licence, hygiene rating, supplier ...)
are filled with a typical restaurant's values; the app lists them under "How this is scored / assumptions".

## Files

| file | what |
|---|---|
| `app.py` | Streamlit app |
| `ingredient_nlp.py` | dish name → ingredients |
| `Food_Adulteration_Risk_classification.ipynb` | training, evaluation, SHAP, menu scoring walkthrough |
| `sus_food_detector_lgbm.pkl` / `.txt` | trained model bundle (with threshold + feature list) / plain LightGBM booster |
| `benchmark_nlp.py`, `benchmark_results.md` | NLP robustness and downstream benchmark |
