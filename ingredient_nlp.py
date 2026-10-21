"""
Dish-name -> ingredient inference for SusFoodDetector.

The LightGBM model scores an item through its `primary_ingredient`, `ingredient_grams_est`,
`veg_nonveg` and `dish_category` columns. A real menu only has dish names, and many names don't
mention what's inside: "Haleem" is mutton + wheat + lentils + ghee, "Dal Makhani" is butter + cream,
"Rasgulla" is chhena (paneer). This module turns a raw menu string into those columns.

Resolution cascade (first hit wins):
  1. Knowledge base   - curated Indian dishes with their ingredients, matched after spelling
                        normalisation ("Dal Makhni", "Hydrabadi Chiken Biriyani") and per-token fuzzy
                        correction. Multi-dish names ("Rabri Jalebi", "Paneer Biryani") are covered by
                        combining several entries and the ingredients named outright.
  2. Keywords         - ingredient words found in the name (paneer, ghee, makhani, kesar, fried ...).
  3. Claude (optional)- for names nothing above recognises; constrained to the same ingredient
                        vocabulary via structured outputs and cached to disk.

Usage:
    from ingredient_nlp import IngredientInferer
    nlp = IngredientInferer()                      # IngredientInferer(use_llm=True) to enable Claude
    nlp.infer("Hyderabadi Haleem (Full)")          # -> DishProfile(...)
    menu = nlp.enrich(menu_df)                     # fills primary_ingredient, veg_nonveg, ...
    rows = expand_tracked_ingredients(menu, loc_ing_stats)   # one row per risky ingredient
"""
import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache

try:
    from rapidfuzz import fuzz, process
except ImportError:  # difflib fallback keeps the module dependency-free
    fuzz = process = None
    import difflib

# --------------------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------------------

# Ingredients the model was trained on (reference market price INR/kg, as in generate_dataset.py)
TRACKED_INGREDIENTS = {
    "paneer": 380, "khoya": 420, "ghee": 650, "milk": 60, "curd": 80, "butter": 550, "soya": 180,
    "red_chilli_powder": 300, "turmeric": 220, "edible_oil": 150, "honey": 500, "saffron": 250000,
    "green_peas": 120, "chicken": 240, "fish": 350, "tea_leaves": 400,
}
# Recognised but not modelled (no adulteration stats for them in the training data)
OTHER_INGREDIENTS = {"mutton", "egg", "cream", "cheese", "wheat", "maida", "rice", "pulses", "besan",
                     "chickpeas", "potato", "vegetables", "sugar", "coconut", "nuts", "maize"}
ALL_INGREDIENTS = set(TRACKED_INGREDIENTS) | OTHER_INGREDIENTS
MEAT = {"chicken", "mutton", "fish", "egg"}
PROTEINS = MEAT | {"paneer", "soya"}
CATEGORIES = ["main", "starter", "rice", "snack", "bread", "dessert", "beverage", "side"]

# --------------------------------------------------------------------------------------
# Knowledge base: (dish, category, "ingredient:grams ...", "alias|alias")
# The first tracked ingredient is the primary one (matches the training data for the dishes it has).
# Chhena is listed as paneer and prawns/crab as fish, since those share the same adulteration risks.
# --------------------------------------------------------------------------------------
DISH_KB = [
    # --- dishes present in the training data ---
    ("Paneer Butter Masala", "main", "paneer:150 butter:15 cream:20", "paneer makhani|butter paneer"),
    ("Kadai Paneer", "main", "paneer:150", "kadhai paneer|karahi paneer"),
    ("Palak Paneer", "main", "paneer:120 vegetables:100", "saag paneer"),
    ("Shahi Paneer", "main", "paneer:150 cream:30 nuts:15", ""),
    ("Matar Paneer", "main", "paneer:120 green_peas:60", ""),
    ("Paneer Tikka", "starter", "paneer:200 curd:40", ""),
    ("Paneer Bhurji", "main", "paneer:150", ""),
    ("Chilli Paneer", "starter", "paneer:150 edible_oil:20", ""),
    ("Paneer Fried Rice", "rice", "paneer:80 rice:150", ""),
    ("Paneer Kathi Roll", "snack", "paneer:80 maida:60", "paneer roll"),
    ("Paneer Paratha", "bread", "paneer:80 wheat:80 ghee:10", ""),
    ("Paneer Pakora", "snack", "paneer:100 besan:40 edible_oil:30", "paneer pakoda"),
    ("Paneer Dosa", "main", "paneer:80 rice:80 pulses:30", ""),
    ("Soya Chaap Masala", "main", "soya:200 cream:20 butter:10", "soya chaap|chaap masala"),
    ("Malai Soya Chaap", "starter", "soya:200 cream:30 curd:30", "malai chaap"),
    ("Tandoori Soya Chaap", "starter", "soya:200 curd:40", "tandoori chaap"),
    ("Soya Chunks Curry", "main", "soya:100", "soya chunks|nutrela curry"),
    ("Soya Keema", "main", "soya:120", ""),
    ("Chilli Soya", "starter", "soya:120 edible_oil:20", ""),
    ("Soya Biryani", "rice", "soya:100 rice:150 ghee:15", ""),
    ("Soya Chaap Roll", "snack", "soya:120 maida:60", "chaap roll"),
    ("Malai Kofta", "main", "khoya:60 paneer:60 cream:30 edible_oil:20", ""),
    ("Khoya Matar", "main", "khoya:80 green_peas:60", "mawa matar"),
    ("Gulab Jamun", "dessert", "khoya:60 edible_oil:30 sugar:60", "gulab jamoon|gulabjamun"),
    ("Kalakand", "dessert", "khoya:100 milk:100", ""),
    ("Khoya Barfi", "dessert", "khoya:100 sugar:40", "mawa barfi|barfi"),
    ("Gajar Halwa", "dessert", "khoya:50 ghee:20 milk:150", "gajar ka halwa|carrot halwa"),
    ("Peda", "dessert", "khoya:80 sugar:30", "pedha"),
    ("Ghee Roast Dosa", "main", "ghee:25 rice:100 pulses:30", "ghee roast"),
    ("Ghee Pongal", "main", "ghee:25 rice:100 pulses:40", "ven pongal|pongal"),
    ("Desi Ghee Paratha", "bread", "ghee:20 wheat:80", "ghee paratha"),
    ("Mysore Pak", "dessert", "ghee:50 besan:40 sugar:50", ""),
    ("Moong Dal Halwa", "dessert", "ghee:60 pulses:60 khoya:20", ""),
    ("Ghee Rice", "rice", "ghee:30 rice:150", "neichoru"),
    ("Dal Bati Churma", "main", "ghee:50 wheat:120 pulses:60", "dal baati"),
    ("Besan Ladoo", "dessert", "ghee:40 besan:50 sugar:40", ""),
    ("Kheer", "dessert", "milk:200 rice:30", "chawal ki kheer|rice kheer"),
    ("Rasmalai", "dessert", "milk:150 paneer:60", "ras malai"),
    ("Rabri", "dessert", "milk:250 sugar:30", "rabdi"),
    ("Cold Coffee", "beverage", "milk:200", ""),
    ("Badam Milk", "beverage", "milk:250 nuts:15", ""),
    ("Filter Coffee", "beverage", "milk:100", "kaapi"),
    ("Mango Milkshake", "beverage", "milk:200", "mango shake"),
    ("Sweet Lassi", "beverage", "curd:250", "lassi"),
    ("Dahi Vada", "snack", "curd:120 pulses:50 edible_oil:20", "dahi bhalla"),
    ("Curd Rice", "rice", "curd:150 rice:100", "thayir sadam|dahi chawal"),
    ("Boondi Raita", "side", "curd:120 besan:15", "raita"),
    ("Butter Chicken", "main", "butter:30 chicken:200 cream:30", "chicken makhani|chicken makhanwala"),
    ("Pav Bhaji", "snack", "butter:25 vegetables:150", ""),
    ("Butter Naan", "bread", "butter:10 maida:80", ""),
    ("Dal Makhani", "main", "butter:25 pulses:80 cream:20", "maa ki dal|kaali dal|dal bukhara"),
    ("Chicken 65", "starter", "red_chilli_powder:8 chicken:200 edible_oil:30", ""),
    ("Laal Maas", "main", "red_chilli_powder:10 mutton:250 ghee:20", "lal maas"),
    ("Mirchi Bajji", "snack", "red_chilli_powder:5 besan:40 edible_oil:30", "mirchi bhajji|mirchi bhaji|mirchi vada"),
    ("Schezwan Fried Rice", "rice", "red_chilli_powder:6 rice:150 edible_oil:20", "szechuan fried rice"),
    ("Chilli Chicken", "starter", "red_chilli_powder:6 chicken:180 edible_oil:20", ""),
    ("Misal Pav", "snack", "red_chilli_powder:8 pulses:80", "misal"),
    ("Dal Tadka", "main", "turmeric:3 pulses:80 ghee:10", "dal fry|yellow dal|dal"),
    ("Sambar", "side", "turmeric:3 pulses:50 vegetables:80", "sambhar"),
    ("Haldi Doodh", "beverage", "turmeric:5 milk:200", "haldi milk|turmeric latte|golden milk|turmeric milk"),
    ("Aloo Gobi", "main", "turmeric:3 potato:100 vegetables:100", ""),
    ("Khichdi", "main", "turmeric:3 rice:80 pulses:50 ghee:10", "khichri"),
    ("Samosa", "snack", "edible_oil:30 maida:50 potato:60", ""),
    ("Vada Pav", "snack", "edible_oil:25 besan:30 potato:60", ""),
    ("Chole Bhature", "main", "edible_oil:40 chickpeas:100 maida:80", "chole bhatura|chole bhatoore"),
    ("Onion Pakora", "snack", "edible_oil:40 besan:40", "onion bhaji|pyaz pakora|kanda bhaji|pakora"),
    ("French Fries", "snack", "edible_oil:40 potato:150", "fries|finger chips"),
    ("Jalebi", "dessert", "edible_oil:40 maida:50 sugar:50", "jilebi"),
    ("Medu Vada", "snack", "edible_oil:30 pulses:60", "uzhunnu vada|vada"),
    ("Puri Sabzi", "main", "edible_oil:35 wheat:60 potato:80", "puri bhaji|aloo puri"),
    ("Kachori", "snack", "edible_oil:30 maida:50 pulses:30", ""),
    ("Honey Chilli Potato", "starter", "honey:20 potato:150 edible_oil:30", ""),
    ("Honey Lemon Tea", "beverage", "honey:15 tea_leaves:2", "honey lemon"),
    ("Honey Chilli Cauliflower", "starter", "honey:20 vegetables:150 edible_oil:30", "honey chilli gobi"),
    ("Kesar Kulfi", "dessert", "saffron:0.05 milk:150", ""),
    ("Hyderabadi Dum Biryani", "rice", "saffron:0.10 mutton:180 rice:150 ghee:20", "dum biryani"),
    ("Kesar Pista Shake", "beverage", "saffron:0.05 milk:200 nuts:10", ""),
    ("Zafrani Pulao", "rice", "saffron:0.10 rice:150 ghee:15", "kesar pulao"),
    ("Kesar Peda", "dessert", "saffron:0.03 khoya:80", ""),
    ("Matar Pulao", "rice", "green_peas:60 rice:150 ghee:10", ""),
    ("Aloo Matar", "main", "green_peas:60 potato:100", ""),
    ("Veg Fried Rice", "rice", "green_peas:40 rice:150 edible_oil:15", "fried rice"),
    ("Chicken Biryani", "rice", "chicken:180 rice:150 ghee:15", "biryani"),
    ("Tandoori Chicken", "starter", "chicken:300 curd:40 butter:10", "tangdi kebab"),
    ("Chicken Curry", "main", "chicken:200 edible_oil:15", "chicken masala"),
    ("Chicken Shawarma", "snack", "chicken:120 maida:60", "shawarma"),
    ("Chicken Momos", "snack", "chicken:80 maida:60", ""),
    ("Fish Fry", "starter", "fish:200 edible_oil:30", ""),
    ("Fish Curry", "main", "fish:200 coconut:40", "meen curry"),
    ("Amritsari Fish", "starter", "fish:200 besan:30 edible_oil:30", ""),
    ("Masala Chai", "beverage", "tea_leaves:3 milk:100", "chai"),
    ("Cutting Chai", "beverage", "tea_leaves:2 milk:60", ""),
    ("Iced Tea", "beverage", "tea_leaves:4", ""),
    # --- dishes whose names hide what's inside ---
    ("Haleem", "main", "mutton:150 wheat:60 pulses:40 ghee:30", ""),
    ("Chicken Haleem", "main", "chicken:150 wheat:60 pulses:40 ghee:30", ""),
    ("Nihari", "main", "mutton:200 ghee:30 wheat:15", "nalli nihari"),
    ("Paya", "main", "mutton:200 edible_oil:20", "paya soup|trotters"),
    ("Rogan Josh", "main", "mutton:200 curd:40 red_chilli_powder:5 ghee:15", "mutton rogan josh|roganjosh"),
    ("Mutton Curry", "main", "mutton:200 edible_oil:20", "mutton masala"),
    ("Mutton Korma", "main", "mutton:200 curd:40 ghee:20 nuts:15", ""),
    ("Chicken Korma", "main", "chicken:200 curd:40 ghee:20 nuts:15", ""),
    ("Mutton Biryani", "rice", "mutton:180 rice:150 ghee:20 saffron:0.03", ""),
    ("Keema", "main", "mutton:150", "mutton keema|kheema"),
    ("Seekh Kebab", "starter", "mutton:200 ghee:10", "mutton seekh kebab"),
    ("Chicken Seekh Kebab", "starter", "chicken:200 edible_oil:10", ""),
    ("Galouti Kebab", "starter", "mutton:150 ghee:20", "galawati kebab"),
    ("Shami Kebab", "starter", "mutton:120 pulses:30 edible_oil:20", ""),
    ("Chicken Tikka", "starter", "chicken:250 curd:40 butter:10", ""),
    ("Chicken Tikka Masala", "main", "chicken:200 butter:15 cream:30 curd:20", ""),
    ("Malai Tikka", "starter", "chicken:250 cream:40 curd:30", "chicken malai tikka|reshmi kebab|afghani chicken"),
    ("Kadai Chicken", "main", "chicken:200 edible_oil:20", "chicken kadai|karahi chicken"),
    ("Chicken Lollipop", "starter", "chicken:200 edible_oil:30 red_chilli_powder:5", ""),
    ("Chicken Manchurian", "starter", "chicken:150 edible_oil:20", ""),
    ("Egg Curry", "main", "egg:120 edible_oil:15", ""),
    ("Egg Bhurji", "main", "egg:150 butter:10", ""),
    ("Omelette", "snack", "egg:120 butter:5", "omelet|masala omelette"),
    ("Prawn Curry", "main", "fish:200 coconut:40", "prawn masala"),
    ("Fish Tikka", "starter", "fish:200 curd:40", ""),
    ("Paneer Tikka Masala", "main", "paneer:150 butter:10 cream:30 curd:20", ""),
    ("Paneer Lababdar", "main", "paneer:150 butter:15 cream:30", ""),
    ("Navratan Korma", "main", "vegetables:150 cream:40 paneer:30 nuts:15", ""),
    ("Methi Malai Matar", "main", "green_peas:80 cream:40", "methi matar malai"),
    ("Dum Aloo", "main", "potato:150 curd:40", "kashmiri dum aloo"),
    ("Rajma", "main", "pulses:100 butter:5", "rajma masala"),
    ("Chole", "main", "chickpeas:120 edible_oil:15", "chana masala|pindi chole"),
    ("Sarson Ka Saag", "main", "vegetables:200 butter:20 ghee:10", "sarson da saag|saag"),
    ("Kadhi Pakora", "main", "curd:200 besan:40 edible_oil:20", "kadhi|punjabi kadhi"),
    ("Baingan Bharta", "main", "vegetables:200 edible_oil:15", ""),
    ("Bhindi Masala", "main", "vegetables:150 edible_oil:20", "bhindi fry"),
    ("Mix Veg", "main", "vegetables:200", "mixed vegetable"),
    ("Veg Manchurian", "starter", "vegetables:150 maida:30 edible_oil:30", "gobi manchurian|manchurian"),
    ("Hakka Noodles", "main", "maida:150 edible_oil:20 vegetables:80", "noodles|chowmein|chow mein"),
    ("Spring Roll", "snack", "maida:40 vegetables:60 edible_oil:30", ""),
    ("Veg Momos", "snack", "maida:60 vegetables:60", "momos"),
    ("Hara Bhara Kebab", "starter", "vegetables:100 potato:50 edible_oil:20", ""),
    ("Dahi Kebab", "starter", "curd:150 edible_oil:20", "dahi ke kebab"),
    ("Masala Dosa", "main", "edible_oil:15 rice:100 pulses:40 potato:80", "dosa|plain dosa"),
    ("Idli", "main", "rice:80 pulses:30", ""),
    ("Uttapam", "main", "edible_oil:10 rice:100 pulses:30 vegetables:40", "uthappam"),
    ("Upma", "main", "edible_oil:10 wheat:80 ghee:5", "rava upma"),
    ("Rava Kesari", "dessert", "ghee:30 wheat:60 sugar:50 saffron:0.02", "kesari bath"),
    ("Bisi Bele Bath", "rice", "ghee:20 rice:100 pulses:50", "bisibele bath"),
    ("Avial", "side", "curd:40 vegetables:150 coconut:40", "aviyal"),
    ("Mor Kuzhambu", "side", "curd:150 coconut:30", ""),
    ("Pani Puri", "snack", "edible_oil:15 pulses:20 potato:30", "golgappa|gol gappa|puchka"),
    ("Bhel Puri", "snack", "edible_oil:10", "bhel"),
    ("Sev Puri", "snack", "edible_oil:20 potato:30", ""),
    ("Dahi Puri", "snack", "curd:80 edible_oil:15 potato:30", ""),
    ("Papdi Chaat", "snack", "curd:80 edible_oil:20", "papri chaat|chaat"),
    ("Aloo Tikki", "snack", "edible_oil:25 potato:120", ""),
    ("Raj Kachori", "snack", "curd:80 edible_oil:30 pulses:30", ""),
    ("Dabeli", "snack", "butter:15 potato:80", ""),
    ("Bread Pakora", "snack", "edible_oil:40 besan:30 wheat:50", ""),
    ("Aloo Paratha", "bread", "butter:10 wheat:80 potato:80", ""),
    ("Plain Paratha", "bread", "ghee:10 wheat:80", "paratha|tawa paratha"),
    ("Naan", "bread", "maida:80", "plain naan"),
    ("Garlic Naan", "bread", "butter:10 maida:80", "butter garlic naan"),
    ("Roti", "bread", "wheat:80", "tandoori roti|chapati|phulka"),
    ("Laccha Paratha", "bread", "ghee:15 wheat:80", "lachha paratha"),
    ("Kulcha", "bread", "butter:10 maida:80", "amritsari kulcha"),
    ("Bhatura", "bread", "edible_oil:30 maida:80", ""),
    ("Jeera Rice", "rice", "ghee:10 rice:150", ""),
    ("Veg Pulao", "rice", "ghee:10 rice:150 vegetables:60", "pulao"),
    ("Veg Biryani", "rice", "ghee:15 saffron:0.03 rice:150 vegetables:100", ""),
    ("Rasgulla", "dessert", "paneer:80 sugar:60", "rasagola|rosogolla|roshogolla"),
    ("Cham Cham", "dessert", "paneer:80 khoya:20 sugar:40", ""),
    ("Sandesh", "dessert", "paneer:80 sugar:30", ""),
    ("Mishti Doi", "dessert", "curd:150 milk:100", ""),
    ("Shrikhand", "dessert", "curd:150 saffron:0.02 sugar:40", ""),
    ("Basundi", "dessert", "milk:250 sugar:30", ""),
    ("Payasam", "dessert", "milk:200 ghee:10", "paal payasam|semiya payasam"),
    ("Phirni", "dessert", "milk:200 rice:20", "firni"),
    ("Malai Kulfi", "dessert", "milk:150", "kulfi"),
    ("Falooda", "beverage", "milk:200", "royal falooda"),
    ("Thandai", "beverage", "milk:200 nuts:15 saffron:0.02", ""),
    ("Shahi Tukda", "dessert", "milk:150 ghee:30 wheat:50 saffron:0.02", "shahi tukra|double ka meetha"),
    ("Sheer Khurma", "dessert", "milk:250 ghee:15 nuts:20", "sheer korma"),
    ("Sooji Halwa", "dessert", "ghee:30 wheat:50 sugar:40", "suji halwa|sheera|halwa"),
    ("Motichoor Ladoo", "dessert", "ghee:30 besan:40 sugar:40", "ladoo"),
    ("Kaju Katli", "dessert", "ghee:5 nuts:80 sugar:40", "kaju barfi"),
    ("Soan Papdi", "dessert", "ghee:30 besan:30 sugar:40", ""),
    ("Ghevar", "dessert", "ghee:40 maida:50 milk:30", ""),
    ("Malpua", "dessert", "ghee:30 milk:50 maida:40", ""),
    ("Imarti", "dessert", "ghee:30 pulses:40 sugar:40", "amriti"),
    ("Balushahi", "dessert", "ghee:40 maida:50", ""),
    ("Kala Jamun", "dessert", "khoya:60 edible_oil:30 sugar:60", ""),
    ("Ginger Tea", "beverage", "tea_leaves:3 milk:100", "adrak chai"),
    ("Green Tea", "beverage", "tea_leaves:2", ""),
    ("Cappuccino", "beverage", "milk:150", "latte|cafe latte|coffee"),
    ("Hot Chocolate", "beverage", "milk:200", ""),
    ("Buttermilk", "beverage", "curd:150", "chaas|masala chaas|mattha|neer mor"),
    ("Mango Lassi", "beverage", "curd:200", ""),
    ("Milkshake", "beverage", "milk:200", "shake"),
]

# --------------------------------------------------------------------------------------
# Text normalisation
# --------------------------------------------------------------------------------------

# Spelling / transliteration variants -> one canonical token (applied to menu text and KB aliases alike)
VARIANTS = {
    "makhni": "makhani", "makhanwala": "makhani", "makkhani": "makhani", "makkhan": "butter",
    "daal": "dal", "dhal": "dal", "panir": "paneer", "paner": "paneer", "panner": "paneer",
    "biriyani": "biryani", "biriani": "biryani", "briyani": "biryani", "biryaani": "biryani",
    "chiken": "chicken", "chikken": "chicken", "murg": "chicken", "murgh": "chicken",
    "gosht": "mutton", "ghosht": "mutton", "lamb": "mutton", "goat": "mutton",
    "machli": "fish", "machhi": "fish", "macchi": "fish", "meen": "fish",
    "jhinga": "prawn", "prawns": "prawn", "shrimp": "prawn", "shrimps": "prawn",
    "anda": "egg", "eggs": "egg", "dahi": "curd", "yogurt": "curd", "yoghurt": "curd",
    "doodh": "milk", "mattar": "matar", "mutter": "matar", "peas": "matar",
    "gobhi": "gobi", "alu": "aloo", "zafran": "kesar", "saffron": "kesar", "ghi": "ghee",
    "halim": "haleem", "nahari": "nihari", "tika": "tikka", "masaala": "masala",
    "kabab": "kebab", "kebabs": "kebab", "kababs": "kebab", "kofte": "kofta",
    "chaay": "chai", "chaai": "chai", "tea": "chai", "milkshake": "shake",
    "laddu": "ladoo", "laddoo": "ladoo",
    "pakoda": "pakora", "pakode": "pakora", "pakore": "pakora", "pakoras": "pakora",
    "bhatoore": "bhature", "chhole": "chole", "channa": "chana", "poori": "puri", "wada": "vada",
    "idly": "idli", "momo": "momos", "kulfee": "kulfi", "rabdi": "rabri", "jamoon": "jamun",
    "chapathi": "chapati", "chappati": "chapati", "parantha": "paratha",
    "szechwan": "schezwan", "szechuan": "schezwan", "chilly": "chilli", "chili": "chilli",
    "kadhai": "kadai", "karahi": "kadai", "mawa": "khoya", "khoa": "khoya",
    "cottage": "paneer", "veggie": "veg", "vegetable": "veg", "vegetables": "veg",
}
# Menu filler that says nothing about ingredients
NOISE = {"half", "full", "plate", "quarter", "regular", "large", "small", "medium", "special", "spl",
         "house", "chef", "chefs", "signature", "homestyle", "home", "style", "combo", "ml", "gm", "g",
         "kg", "pcs", "pc", "piece", "pieces", "with", "and", "n", "of", "the", "a", "in", "new",
         "fresh", "hot", "classic", "premium", "royal", "deluxe", "desi", "pure", "original", "famous",
         "hyderabadi", "lucknowi", "punjabi", "amritsari", "delhi", "mumbai", "bombay", "kolkata",
         "awadhi", "mughlai", "nawabi", "kashmiri", "kerala", "chettinad", "andhra", "rajasthani",
         "dry", "gravy", "boneless", "bone", "portion", "serve", "serves"}

# Ingredient words found directly in a name: token -> (ingredient or None, strong?)
# Strong = the dish is named after it ("Paneer ...", "Kesar ..."); weak = a cooking-method hint.
# `None` with "veg" means "no meat" and strips the default protein of a generic dish.
KEYWORDS = {
    "paneer": ("paneer", True), "chhena": ("paneer", True), "khoya": ("khoya", True),
    "ghee": ("ghee", True), "butter": ("butter", True), "makhani": ("butter", True),
    "malai": ("cream", True), "cream": ("cream", True), "cheese": ("cheese", True),
    "milk": ("milk", True), "curd": ("curd", True), "raita": ("curd", True), "lassi": ("curd", True),
    "chaas": ("curd", True), "soya": ("soya", True), "chaap": ("soya", True), "honey": ("honey", True),
    "kesar": ("saffron", True), "haldi": ("turmeric", True), "turmeric": ("turmeric", True),
    "chicken": ("chicken", True), "fish": ("fish", True), "prawn": ("fish", True), "crab": ("fish", True),
    "mutton": ("mutton", True), "keema": ("mutton", True), "egg": ("egg", True),
    "matar": ("green_peas", True), "chai": ("tea_leaves", True), "coffee": ("milk", False),
    "shake": ("milk", False), "kulfi": ("milk", False), "kheer": ("milk", False),
    "halwa": ("ghee", False), "ladoo": ("ghee", False), "barfi": ("khoya", False),
    "mirchi": ("red_chilli_powder", False), "chilli": ("red_chilli_powder", False),
    "schezwan": ("red_chilli_powder", False), "tandoori": ("curd", False),
    "fried": ("edible_oil", False), "fry": ("edible_oil", False), "pakora": ("edible_oil", False),
    "bhaji": ("edible_oil", False), "bajji": ("edible_oil", False), "vada": ("edible_oil", False),
    "samosa": ("edible_oil", False), "puri": ("edible_oil", False), "bhature": ("edible_oil", False),
    "fries": ("edible_oil", False), "crispy": ("edible_oil", False), "kachori": ("edible_oil", False),
    "tempura": ("edible_oil", False), "cutlet": ("edible_oil", False),
    "rice": ("rice", False), "chawal": ("rice", False), "dal": ("pulses", False),
    "rajma": ("pulses", False), "chana": ("chickpeas", False), "aloo": ("potato", False),
    "veg": (None, True),
}
CATEGORY_HINTS = {
    "biryani": "rice", "pulao": "rice", "rice": "rice", "chawal": "rice", "khichdi": "rice",
    "naan": "bread", "roti": "bread", "paratha": "bread", "kulcha": "bread", "chapati": "bread",
    "halwa": "dessert", "ladoo": "dessert", "barfi": "dessert", "kheer": "dessert", "kulfi": "dessert",
    "jamun": "dessert", "payasam": "dessert", "sweet": "dessert", "mithai": "dessert",
    "chai": "beverage", "coffee": "beverage", "shake": "beverage", "lassi": "beverage",
    "juice": "beverage", "smoothie": "beverage", "soda": "beverage",
    "tikka": "starter", "kebab": "starter", "65": "starter", "lollipop": "starter", "fry": "starter",
    "roll": "snack", "momos": "snack", "samosa": "snack", "chaat": "snack", "pakora": "snack",
    "sandwich": "snack", "burger": "snack", "puri": "snack", "raita": "side", "salad": "side",
    "curry": "main", "masala": "main", "korma": "main", "kadai": "main", "thali": "main",
}


def _tokens(text):
    text = text.lower().replace("&", " ")
    text = re.sub(r"\(.*?\)|\[.*?\]", " ", text)           # "(Full)", "[Chef's Special]"
    text = re.sub(r"\b\d+\s*(ml|gm|g|kg|pcs?|pieces?)\b", " ", text)
    toks = re.findall(r"[a-z]+|\b65\b", text)
    return [VARIANTS.get(t, t) for t in toks]


def _spec(spec):
    out = {}
    for part in spec.split():
        ing, grams = part.split(":")
        assert ing in ALL_INGREDIENTS, f"unknown ingredient {ing!r} in DISH_KB"
        out[ing] = float(grams)
    return out


@dataclass
class DishProfile:
    query: str
    source: str                       # kb | kb_partial | kb_fuzzy | keywords | llm | unknown
    confidence: float
    matched_dishes: list = field(default_factory=list)
    ingredients: dict = field(default_factory=dict)   # ingredient -> grams (first tracked = primary)
    dish_category: str = None
    veg_nonveg: str = None

    @property
    def tracked_ingredients(self):
        return {k: v for k, v in self.ingredients.items() if k in TRACKED_INGREDIENTS}

    @property
    def primary_ingredient(self):
        return next(iter(self.tracked_ingredients), None)

    @property
    def primary_grams(self):
        return self.tracked_ingredients.get(self.primary_ingredient)


def _profile(query, source, conf, dishes, ings, category):
    veg = None if not ings else ("nonveg" if MEAT & set(ings) else "veg")
    return DishProfile(query, source, round(conf, 3), dishes, ings, category, veg)


class IngredientInferer:
    def __init__(self, use_llm=False, llm_model="claude-opus-5-5", cache_path="ingredient_llm_cache.json",
                 fuzzy_threshold=86):
        self.use_llm = use_llm
        self.llm_model = llm_model
        self.cache_path = cache_path
        self.fuzzy_threshold = fuzzy_threshold
        self._llm_cache = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path) as f:
                self._llm_cache = json.load(f)
        self._client = None

        # alias (normalised token tuple) -> (dish, category, ingredients)
        self.kb = {}
        for dish, cat, spec, aliases in DISH_KB:
            entry = (dish, cat, _spec(spec))
            for name in [dish] + [a for a in aliases.split("|") if a]:
                self.kb.setdefault(tuple(_tokens(name)), entry)
        self._alias_strings = {" ".join(k): k for k in self.kb}
        self._vocab = {t for k in self.kb for t in k} | set(KEYWORDS) | set(CATEGORY_HINTS)

    # ---- matching helpers ----
    def _correct(self, tok):
        """Snap a misspelt token to the vocabulary: 1 edit for words up to 7 letters, 2 beyond.

        Edits are insertions, deletions, substitutions and adjacent swaps ("Jlaebi" -> "jalebi").
        Ties go to the same-length word, then the closer string overall.
        """
        if tok in self._vocab or len(tok) < 4:
            return tok
        max_d = 1 if len(tok) <= 7 else 2
        best = None
        for v in self._vocab:
            if abs(len(v) - len(tok)) > max_d:
                continue
            d = _osa_distance(tok, v, max_d)
            if d <= max_d:
                rank = (d, len(v) != len(tok), -_similarity(tok, v), v)
                best = min(best, rank) if best else rank
        return best[-1] if best else tok

    def _clean(self, name):
        toks = [self._correct(t) for t in _tokens(name)]
        # drop filler words unless they're part of a known dish name ("Hyderabadi Dum Biryani")
        kept = [t for t in toks if t not in NOISE or any(t in k for k in self.kb)]
        return kept or toks

    def _cover(self, toks):
        """Greedily cover the name with KB aliases, longest first (aliases match as token sets)."""
        remaining, found = list(toks), []
        for alias in sorted(self.kb, key=len, reverse=True):
            if set(alias) <= set(remaining):
                found.append((min(toks.index(t) for t in alias), alias))
                for t in alias:
                    remaining.remove(t)
        found.sort()
        return [a for _, a in found], remaining

    def _keywords(self, toks):
        strong, weak = {}, {}
        for t in toks:
            ing, is_strong = KEYWORDS.get(t, (None, None))
            if ing:
                (strong if is_strong else weak).setdefault(ing, None)
        return strong, weak

    def _category(self, toks, dishes_by_pos):
        for t in reversed(toks):                  # head noun is usually last: "Mix Veg Raita"
            for alias, cat in dishes_by_pos:
                if t in alias:
                    return cat
            if t in CATEGORY_HINTS:
                return CATEGORY_HINTS[t]
        return dishes_by_pos[0][1] if dishes_by_pos else "main"

    # ---- public API ----
    def infer(self, name):
        toks = self._clean(str(name))
        if not toks:
            return _profile(name, "unknown", 0.0, [], {}, None)

        key = tuple(toks)
        if key in self.kb:
            dish, cat, ings = self.kb[key]
            return _profile(name, "kb", 1.0, [dish], dict(ings), cat)

        aliases, leftover = self._cover(toks)
        source = "kb_partial"
        if not aliases:
            # whole-string fuzzy match catches spellings the token pass missed
            q = " ".join(toks)
            if process:
                hit = process.extractOne(q, list(self._alias_strings), scorer=fuzz.ratio,
                                         score_cutoff=self.fuzzy_threshold)
                hit = hit[0] if hit else None
            else:
                hit = next(iter(difflib.get_close_matches(q, list(self._alias_strings), n=1,
                                                          cutoff=self.fuzzy_threshold / 100)), None)
            if hit:
                dish, cat, ings = self.kb[self._alias_strings[hit]]
                # reject a lookalike that contradicts an ingredient the name states
                # ("Butter Garlic Prawns" is not "Butter Garlic Naan")
                named = {KEYWORDS[t][0] for t in toks if t in KEYWORDS and KEYWORDS[t][1] and KEYWORDS[t][0]}
                if named <= set(ings):
                    return _profile(name, "kb_fuzzy", 0.85, [dish], dict(ings), cat)

        strong, weak = self._keywords(leftover)
        if not aliases and not strong and not weak:
            return self._llm_one(name) if self.use_llm else _profile(name, "unknown", 0.0, [], {}, None)

        # ingredients named outright come first ("Paneer Biryani" -> paneer is the primary)
        ings = {k: None for k in strong if k}
        named_proteins = (set(strong) & PROTEINS) | ({"veg"} if "veg" in leftover else set())
        dishes = []
        for alias in aliases:
            dish, cat, dish_ings = self.kb[alias]
            # a named protein replaces a generic dish's default one ("Egg Biryani", "Veg Haleem")
            dropped = {i for i in dish_ings if named_proteins and i in PROTEINS and i not in strong}
            if dropped:
                swap = ", ".join(sorted(p for p in named_proteins if p != "veg")) or "veg"
                dish = f"{dish} ({swap} instead of {', '.join(sorted(dropped))})"
            dishes.append(dish)
            for ing, g in dish_ings.items():
                if ing in dropped:
                    continue
                if ings.get(ing) is None:
                    ings[ing] = g
        for k in weak:
            ings.setdefault(k, None)
        ings = {k: (g if g is not None else _default_grams(k)) for k, g in ings.items()}

        cat = self._category(toks, [(a, self.kb[a][1]) for a in aliases])
        if aliases:
            unexplained = [t for t in leftover if t not in NOISE and t not in KEYWORDS]
            coverage = 1 - len(unexplained) / len(toks)
            return _profile(name, source, 0.6 + 0.35 * coverage, dishes, ings, cat)
        return _profile(name, "keywords", 0.6 if strong else 0.4, [], ings, cat)

    def infer_many(self, names):
        names = list(names)
        profiles = {n: self.infer(n) for n in dict.fromkeys(names)}
        if self.use_llm:
            unknown = [n for n, p in profiles.items() if p.source == "unknown" and n not in self._llm_cache]
            if unknown:
                self._llm_batch(unknown)
            for n, p in profiles.items():
                if p.source == "unknown" and n in self._llm_cache:
                    profiles[n] = self._from_llm(n, self._llm_cache[n])
        return [profiles[n] for n in names]

    def enrich(self, df, name_col="dish_name", overwrite=False):
        """Add NLP columns to a menu DataFrame and fill the model's ingredient columns.

        With overwrite=False, existing non-null values of primary_ingredient etc. are kept.
        """
        out = df.copy()
        profiles = self.infer_many(out[name_col].astype(str))
        inferred = {
            "primary_ingredient": [p.primary_ingredient for p in profiles],
            "ingredient_grams_est": [p.primary_grams for p in profiles],
            "veg_nonveg": [p.veg_nonveg for p in profiles],
            "dish_category": [p.dish_category for p in profiles],
        }
        for col, vals in inferred.items():
            if overwrite or col not in out:
                out[col] = vals
            else:
                out[col] = out[col].where(out[col].notna(), vals)
        out["nlp_ingredients"] = [";".join(p.ingredients) for p in profiles]
        out["nlp_tracked_ingredients"] = [";".join(f"{k}:{g:g}" for k, g in p.tracked_ingredients.items())
                                          for p in profiles]
        out["nlp_matched_dish"] = [" + ".join(p.matched_dishes) for p in profiles]
        out["nlp_source"] = [p.source for p in profiles]
        out["nlp_confidence"] = [p.confidence for p in profiles]
        return out

    # ---- Claude fallback ----
    def _llm_one(self, name):
        if name not in self._llm_cache:
            self._llm_batch([name])
        if name in self._llm_cache:
            return self._from_llm(name, self._llm_cache[name])
        return _profile(name, "unknown", 0.0, [], {}, None)

    def _from_llm(self, name, rec):
        ings = {}
        # tracked ingredients first so the primary is a modelled one
        for item in sorted(rec["ingredients"], key=lambda i: i["name"] not in TRACKED_INGREDIENTS):
            if item["name"] in ALL_INGREDIENTS:
                ings.setdefault(item["name"], float(item["grams"]))
        p = _profile(name, "llm", 0.7, [rec.get("dish_identified", "")], ings, rec["dish_category"])
        p.veg_nonveg = rec["veg_nonveg"]   # trust the model here: it may know about meat we don't track
        return p

    def _llm_batch(self, names):
        import anthropic  # only needed when the fallback is used
        if self._client is None:
            self._client = anthropic.Anthropic()
        vocab = sorted(ALL_INGREDIENTS)
        schema = {
            "type": "object",
            "properties": {"dishes": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "menu_text": {"type": "string"},
                    "dish_identified": {"type": "string"},
                    "dish_category": {"type": "string", "enum": CATEGORIES},
                    "veg_nonveg": {"type": "string", "enum": ["veg", "nonveg"]},
                    "ingredients": {"type": "array", "items": {
                        "type": "object",
                        "properties": {"name": {"type": "string", "enum": vocab},
                                       "grams": {"type": "number"}},
                        "required": ["name", "grams"], "additionalProperties": False}},
                },
                "required": ["menu_text", "dish_identified", "dish_category", "veg_nonveg", "ingredients"],
                "additionalProperties": False}}},
            "required": ["dishes"], "additionalProperties": False,
        }
        prompt = (
            "You are helping a food-safety team estimate which adulteration-prone ingredients are in "
            "dishes on Indian restaurant menus. For each menu item below, identify the dish and list "
            "its main ingredients as typically prepared in Indian restaurants, with approximate grams "
            "per single serving. Use only the allowed ingredient names; map chhena to paneer, prawns "
            "and other seafood to fish, lamb/goat to mutton, malai to cream, and skip minor spices and "
            "water. Include cooking fat (ghee, butter or edible_oil) when it's a substantial part of the "
            "dish. Mark veg_nonveg as nonveg if the dish normally contains meat, fish or egg. Return one "
            "entry per item, echoing the menu text exactly.\n\nMenu items:\n"
            + "\n".join(f"- {n}" for n in names)
        )
        resp = self._client.beta.messages.create(
            model=self.llm_model,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": prompt}],
        )
        if resp.stop_reason != "end_turn":
            raise RuntimeError(f"Claude ingredient lookup stopped early: {resp.stop_reason}")
        data = json.loads(next(b.text for b in resp.content if b.type == "text"))
        by_text = {d["menu_text"]: d for d in data["dishes"]}
        for n in names:
            if n in by_text:
                self._llm_cache[n] = by_text[n]
        if self.cache_path:
            with open(self.cache_path, "w") as f:
                json.dump(self._llm_cache, f, indent=1)


@lru_cache(maxsize=None)
def _osa_distance(a, b, max_d):
    """Optimal-string-alignment edit distance, stopping early once it exceeds max_d."""
    prev2, prev = None, list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        if min(cur) > max_d:
            return max_d + 1
        prev2, prev = prev, cur
    return prev[-1]


def _similarity(a, b):
    return fuzz.ratio(a, b) if fuzz else difflib.SequenceMatcher(None, a, b).ratio() * 100


def _default_grams(ing):
    # rough per-serving amount when an ingredient comes only from a keyword
    return {"saffron": 0.05, "red_chilli_powder": 6, "turmeric": 3, "tea_leaves": 3, "edible_oil": 30,
            "ghee": 25, "butter": 15, "honey": 15, "cream": 30, "curd": 100, "milk": 200,
            "green_peas": 60}.get(ing, 120)


def expand_tracked_ingredients(menu, loc_ing_stats=None):
    """One row per (menu item, tracked ingredient) from an `IngredientInferer.enrich` output.

    Sets primary_ingredient / ingredient_grams_est / est_ingredient_cost_inr for each ingredient so
    the item can be scored once per risky ingredient. With `loc_ing_stats` (location_ingredient_stats
    .csv) and a `location_id` column, the loc_ing_* features are re-joined for that ingredient.
    """
    import pandas as pd
    rows = []
    for idx, r in menu.iterrows():
        tracked = [t.split(":") for t in str(r.get("nlp_tracked_ingredients", "")).split(";") if t]
        for i, (ing, grams) in enumerate(tracked):
            row = r.copy()
            row["item_index"] = idx
            row["component"] = i
            row["primary_ingredient"] = ing
            row["ingredient_grams_est"] = float(grams)
            rows.append(row)
    out = pd.DataFrame(rows).reset_index(drop=True)
    if out.empty:
        return out

    if loc_ing_stats is not None and "location_id" in out:
        li = loc_ing_stats.rename(columns={c: f"loc_ing_{c}" for c in loc_ing_stats.columns
                                           if c not in ("location_id", "ingredient")})
        out = out.drop(columns=[c for c in li.columns if c.startswith("loc_ing_") and c in out])
        out = out.merge(li, left_on=["location_id", "primary_ingredient"],
                        right_on=["location_id", "ingredient"], how="left").drop(columns="ingredient")

    price_per_kg = (out["loc_ing_market_price_per_kg"] if "loc_ing_market_price_per_kg" in out
                    else out["primary_ingredient"].map(TRACKED_INGREDIENTS))
    out["est_ingredient_cost_inr"] = (out["ingredient_grams_est"] / 1000 * price_per_kg).round(2)
    if "menu_price_inr" in out:
        out["price_to_ingredient_cost_ratio"] = (
            out["menu_price_inr"] / out["est_ingredient_cost_inr"].clip(lower=0.5)).round(3)
    return out
