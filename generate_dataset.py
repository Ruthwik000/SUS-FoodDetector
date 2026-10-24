#!/usr/bin/env python3
"""
Synthetic dataset generator: restaurant food-adulteration risk.

Idea
----
Food-safety officers can't test every restaurant. But adulteration is not random:
  * some locations have known rackets for specific items (paneer, khoya, ghee, soya ...)
  * restaurants whose menus lean on those items are more exposed
  * an item priced unusually low vs. peers (or vs. the raw ingredient cost) is suspicious
  * licence status, hygiene rating, complaints, supplier type etc. add signal

This script builds a hidden "ground truth" from those mechanisms, then exposes only the
observable features plus the lab-test label, so a model has to learn the relationships.

Stdlib only. Usage:
    python3 generate_dataset.py --restaurants 8000 --seed 42 --out data
"""
import argparse
import csv
import math
import os
import random
import statistics
from collections import defaultdict

# --------------------------------------------------------------------------------------
# Reference tables
# --------------------------------------------------------------------------------------

# name: (national base prevalence, market price INR/kg, cost_sensitivity,
#        sensory detectability, adulterants)
INGREDIENTS = {
    "paneer":            (0.22,    380, 1.0, 0.60, ["starch", "vanaspati_analogue", "detergent", "synthetic_milk"]),
    "khoya":             (0.25,    420, 1.0, 0.40, ["starch", "synthetic_milk_solids", "vanaspati"]),
    "ghee":              (0.18,    650, 1.0, 0.30, ["vanaspati", "animal_body_fat", "palm_oil"]),
    "milk":              (0.15,     60, 0.6, 0.30, ["water", "urea", "detergent", "starch"]),
    "curd":              (0.10,     80, 0.5, 0.25, ["starch", "synthetic_milk"]),
    "butter":            (0.12,    550, 0.8, 0.25, ["vanaspati", "margarine"]),
    "soya":              (0.16,    180, 0.7, 0.40, ["maida_filler", "artificial_colour", "stale_stock"]),
    "red_chilli_powder": (0.14,    300, 0.6, 0.15, ["brick_powder", "sudan_dye", "sawdust"]),
    "turmeric":          (0.12,    220, 0.5, 0.10, ["metanil_yellow", "lead_chromate", "chalk_powder"]),
    "edible_oil":        (0.14,    150, 0.7, 0.50, ["argemone_oil", "reused_oil_high_tpc", "mineral_oil"]),
    "honey":             (0.30,    500, 0.6, 0.15, ["sugar_syrup", "jaggery_syrup"]),
    "saffron":           (0.35, 250000, 1.2, 0.20, ["dyed_corn_silk", "safflower"]),
    "green_peas":        (0.08,    120, 0.3, 0.10, ["malachite_green"]),
    "chicken":           (0.06,    240, 0.4, 0.35, ["formalin", "water_injection", "stale_meat"]),
    "fish":              (0.10,    350, 0.5, 0.35, ["formalin", "ammonia"]),
    "tea_leaves":        (0.12,    400, 0.4, 0.15, ["used_tea_leaves", "coal_tar_dye"]),
}
DAIRY = {"paneer", "khoya", "ghee", "milk", "curd", "butter"}

# cuisine codes: N north_indian, S south_indian, M mughlai, C indo_chinese,
#                F street_food, W sweets, X cafe_beverages
CUISINES = {"N": "north_indian", "S": "south_indian", "M": "mughlai", "C": "indo_chinese",
            "F": "street_food", "W": "sweets", "X": "cafe_beverages"}

# (dish, primary ingredient, grams of that ingredient, base price INR @ casual dining, category, veg, cuisines)
DISHES = [
    ("Paneer Butter Masala", "paneer", 150, 260, "main", "veg", "NM"),
    ("Kadai Paneer", "paneer", 150, 250, "main", "veg", "NM"),
    ("Palak Paneer", "paneer", 120, 240, "main", "veg", "N"),
    ("Shahi Paneer", "paneer", 150, 270, "main", "veg", "NM"),
    ("Matar Paneer", "paneer", 120, 230, "main", "veg", "N"),
    ("Paneer Tikka", "paneer", 200, 280, "starter", "veg", "NMF"),
    ("Paneer Bhurji", "paneer", 150, 220, "main", "veg", "NF"),
    ("Chilli Paneer", "paneer", 150, 240, "starter", "veg", "C"),
    ("Paneer Fried Rice", "paneer", 80, 200, "rice", "veg", "C"),
    ("Paneer Kathi Roll", "paneer", 80, 120, "snack", "veg", "F"),
    ("Paneer Paratha", "paneer", 80, 90, "bread", "veg", "NF"),
    ("Paneer Pakora", "paneer", 100, 140, "snack", "veg", "FN"),
    ("Paneer Dosa", "paneer", 80, 130, "main", "veg", "S"),
    ("Soya Chaap Masala", "soya", 200, 210, "main", "veg", "NM"),
    ("Malai Soya Chaap", "soya", 200, 230, "starter", "veg", "NMF"),
    ("Tandoori Soya Chaap", "soya", 200, 200, "starter", "veg", "NF"),
    ("Soya Chunks Curry", "soya", 100, 150, "main", "veg", "N"),
    ("Soya Keema", "soya", 120, 170, "main", "veg", "N"),
    ("Chilli Soya", "soya", 120, 170, "starter", "veg", "C"),
    ("Soya Biryani", "soya", 100, 180, "rice", "veg", "M"),
    ("Soya Chaap Roll", "soya", 120, 110, "snack", "veg", "F"),
    ("Malai Kofta", "khoya", 60, 250, "main", "veg", "NM"),
    ("Khoya Matar", "khoya", 80, 220, "main", "veg", "N"),
    ("Gulab Jamun", "khoya", 60, 70, "dessert", "veg", "NWMF"),
    ("Kalakand", "khoya", 100, 90, "dessert", "veg", "W"),
    ("Khoya Barfi", "khoya", 100, 80, "dessert", "veg", "W"),
    ("Gajar Halwa", "khoya", 50, 120, "dessert", "veg", "NW"),
    ("Peda", "khoya", 80, 70, "dessert", "veg", "W"),
    ("Ghee Roast Dosa", "ghee", 25, 150, "main", "veg", "S"),
    ("Ghee Pongal", "ghee", 25, 120, "main", "veg", "S"),
    ("Desi Ghee Paratha", "ghee", 20, 60, "bread", "veg", "NF"),
    ("Mysore Pak", "ghee", 50, 90, "dessert", "veg", "WS"),
    ("Moong Dal Halwa", "ghee", 60, 140, "dessert", "veg", "NW"),
    ("Ghee Rice", "ghee", 30, 160, "rice", "veg", "S"),
    ("Dal Bati Churma", "ghee", 50, 220, "main", "veg", "N"),
    ("Besan Ladoo", "ghee", 40, 60, "dessert", "veg", "W"),
    ("Kheer", "milk", 200, 100, "dessert", "veg", "NWM"),
    ("Rasmalai", "milk", 150, 110, "dessert", "veg", "WN"),
    ("Rabri", "milk", 250, 120, "dessert", "veg", "WM"),
    ("Cold Coffee", "milk", 200, 130, "beverage", "veg", "XF"),
    ("Badam Milk", "milk", 250, 80, "beverage", "veg", "XSW"),
    ("Filter Coffee", "milk", 100, 40, "beverage", "veg", "SX"),
    ("Mango Milkshake", "milk", 200, 120, "beverage", "veg", "X"),
    ("Sweet Lassi", "curd", 250, 80, "beverage", "veg", "NXF"),
    ("Dahi Vada", "curd", 120, 90, "snack", "veg", "NF"),
    ("Curd Rice", "curd", 150, 110, "rice", "veg", "S"),
    ("Boondi Raita", "curd", 120, 70, "side", "veg", "NM"),
    ("Butter Chicken", "butter", 30, 320, "main", "nonveg", "NM"),
    ("Pav Bhaji", "butter", 25, 120, "snack", "veg", "F"),
    ("Butter Naan", "butter", 10, 50, "bread", "veg", "NM"),
    ("Dal Makhani", "butter", 25, 220, "main", "veg", "N"),
    ("Chicken 65", "red_chilli_powder", 8, 240, "starter", "nonveg", "SC"),
    ("Laal Maas", "red_chilli_powder", 10, 380, "main", "nonveg", "N"),
    ("Mirchi Bajji", "red_chilli_powder", 5, 50, "snack", "veg", "FS"),
    ("Schezwan Fried Rice", "red_chilli_powder", 6, 180, "rice", "veg", "C"),
    ("Chilli Chicken", "red_chilli_powder", 6, 260, "starter", "nonveg", "C"),
    ("Misal Pav", "red_chilli_powder", 8, 90, "snack", "veg", "F"),
    ("Dal Tadka", "turmeric", 3, 160, "main", "veg", "N"),
    ("Sambar", "turmeric", 3, 60, "side", "veg", "S"),
    ("Haldi Doodh", "turmeric", 5, 60, "beverage", "veg", "X"),
    ("Aloo Gobi", "turmeric", 3, 170, "main", "veg", "N"),
    ("Khichdi", "turmeric", 3, 140, "main", "veg", "NS"),
    ("Samosa", "edible_oil", 30, 25, "snack", "veg", "F"),
    ("Vada Pav", "edible_oil", 25, 30, "snack", "veg", "F"),
    ("Chole Bhature", "edible_oil", 40, 130, "main", "veg", "NF"),
    ("Onion Pakora", "edible_oil", 40, 80, "snack", "veg", "FN"),
    ("French Fries", "edible_oil", 40, 110, "snack", "veg", "FX"),
    ("Jalebi", "edible_oil", 40, 60, "dessert", "veg", "WF"),
    ("Medu Vada", "edible_oil", 30, 60, "snack", "veg", "S"),
    ("Puri Sabzi", "edible_oil", 35, 80, "main", "veg", "NF"),
    ("Kachori", "edible_oil", 30, 30, "snack", "veg", "FW"),
    ("Honey Chilli Potato", "honey", 20, 200, "starter", "veg", "C"),
    ("Honey Lemon Tea", "honey", 15, 70, "beverage", "veg", "X"),
    ("Honey Chilli Cauliflower", "honey", 20, 210, "starter", "veg", "C"),
    ("Kesar Kulfi", "saffron", 0.05, 80, "dessert", "veg", "WF"),
    ("Hyderabadi Dum Biryani", "saffron", 0.10, 300, "rice", "nonveg", "M"),
    ("Kesar Pista Shake", "saffron", 0.05, 150, "beverage", "veg", "X"),
    ("Zafrani Pulao", "saffron", 0.10, 240, "rice", "veg", "M"),
    ("Kesar Peda", "saffron", 0.03, 90, "dessert", "veg", "W"),
    ("Matar Pulao", "green_peas", 60, 170, "rice", "veg", "N"),
    ("Aloo Matar", "green_peas", 60, 160, "main", "veg", "N"),
    ("Veg Fried Rice", "green_peas", 40, 170, "rice", "veg", "C"),
    ("Chicken Biryani", "chicken", 180, 250, "rice", "nonveg", "MS"),
    ("Tandoori Chicken", "chicken", 300, 350, "starter", "nonveg", "NM"),
    ("Chicken Curry", "chicken", 200, 280, "main", "nonveg", "NS"),
    ("Chicken Shawarma", "chicken", 120, 130, "snack", "nonveg", "FM"),
    ("Chicken Momos", "chicken", 80, 120, "snack", "nonveg", "CF"),
    ("Fish Fry", "fish", 200, 300, "starter", "nonveg", "S"),
    ("Fish Curry", "fish", 200, 320, "main", "nonveg", "S"),
    ("Amritsari Fish", "fish", 200, 320, "starter", "nonveg", "N"),
    ("Masala Chai", "tea_leaves", 3, 25, "beverage", "veg", "XFS"),
    ("Cutting Chai", "tea_leaves", 2, 15, "beverage", "veg", "FX"),
    ("Iced Tea", "tea_leaves", 4, 90, "beverage", "veg", "X"),
]

# type: (price multiplier, integrity bias, menu size range, allowed cuisines, share of all restaurants)
RESTAURANT_TYPES = {
    "street_stall":     (0.45, -0.60, (4, 9),   "FCXS",    0.20),
    "dhaba":            (0.60, -0.40, (10, 22), "NMF",     0.14),
    "sweet_shop":       (0.70, -0.20, (8, 18),  "WFX",     0.10),
    "qsr":              (0.80,  0.00, (8, 16),  "FCXN",    0.12),
    "cloud_kitchen":    (0.85, -0.30, (10, 24), "NMCF",    0.13),
    "casual_dining":    (1.00,  0.10, (14, 30), "NSMCX",   0.18),
    "hotel_restaurant": (1.40,  0.35, (18, 34), "NSMCWX",  0.08),
    "fine_dining":      (2.20,  0.60, (14, 26), "NMSW",    0.05),
}

# city: (state, tier, cost index, enforcement intensity, city-level adulteration shift)
CITIES = {
    "Delhi": ("Delhi", 1, 1.15, 1.2, 0.30), "Mumbai": ("Maharashtra", 1, 1.30, 1.1, 0.10),
    "Bengaluru": ("Karnataka", 1, 1.20, 1.0, -0.10), "Hyderabad": ("Telangana", 1, 1.10, 1.0, 0.05),
    "Chennai": ("Tamil Nadu", 1, 1.10, 1.1, -0.15), "Kolkata": ("West Bengal", 1, 1.00, 0.9, 0.10),
    "Pune": ("Maharashtra", 2, 1.05, 1.0, -0.05), "Ahmedabad": ("Gujarat", 2, 1.00, 1.0, 0.00),
    "Jaipur": ("Rajasthan", 2, 0.95, 0.9, 0.20), "Lucknow": ("Uttar Pradesh", 2, 0.90, 0.8, 0.35),
    "Kanpur": ("Uttar Pradesh", 2, 0.85, 0.7, 0.45), "Indore": ("Madhya Pradesh", 2, 0.90, 1.1, 0.10),
    "Bhopal": ("Madhya Pradesh", 2, 0.88, 0.9, 0.15), "Patna": ("Bihar", 2, 0.82, 0.6, 0.45),
    "Chandigarh": ("Chandigarh", 2, 1.05, 1.2, -0.10), "Ludhiana": ("Punjab", 2, 0.95, 0.8, 0.30),
    "Agra": ("Uttar Pradesh", 3, 0.85, 0.7, 0.40), "Varanasi": ("Uttar Pradesh", 3, 0.82, 0.7, 0.35),
    "Nagpur": ("Maharashtra", 2, 0.90, 0.9, 0.05), "Surat": ("Gujarat", 2, 0.95, 0.9, 0.05),
    "Kochi": ("Kerala", 2, 1.00, 1.2, -0.25), "Visakhapatnam": ("Andhra Pradesh", 2, 0.90, 0.9, 0.00),
    "Guwahati": ("Assam", 2, 0.90, 0.7, 0.15), "Bhubaneswar": ("Odisha", 2, 0.88, 0.8, 0.05),
}

# area type: (adulteration logit shift, restaurant density weight)
AREA_TYPES = {
    "railway_station_area": (0.50, 1.3), "highway_stretch": (0.55, 0.8),
    "industrial_belt": (0.40, 0.9), "old_city_market": (0.35, 1.4),
    "residential_suburb": (0.00, 1.0), "university_area": (0.15, 1.1),
    "it_corridor": (-0.25, 1.2), "upscale_commercial": (-0.40, 1.0),
}

SUPPLIERS = {  # name: (logit effect on adulteration, base weight)
    "wholesale_mandi": (0.45, 0.30), "local_vendor": (0.25, 0.30),
    "branded_distributor": (-0.55, 0.25), "own_procurement": (-0.30, 0.15),
}

NAME_A = ["Shree", "New", "Royal", "Sai", "Krishna", "Annapurna", "Spice", "Punjabi", "Delhi",
          "Bombay", "Green", "Golden", "Taj", "Moti", "Ganesh", "Lakshmi", "Urban", "Desi",
          "Classic", "Hari", "Om", "Balaji", "Amrit", "Sagar", "Madras", "Bikaner", "Kwality"]
NAME_B = ["Bhojanalaya", "Dhaba", "Sweets", "Kitchen", "Restaurant", "Cafe", "Foods", "Rasoi",
          "Corner", "Palace", "Grand", "Express", "Bites", "Darbar", "Mithai Bhandar", "Tadka",
          "Point", "Junction", "House", "Treats"]

INTERCEPT = -1.25  # global calibration shift (targets ~15-20% positive items)


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def logit(p):
    return math.log(p / (1.0 - p))


def poisson(rng, lam):
    if lam <= 0:
        return 0
    if lam > 30:  # normal approximation
        return max(0, int(round(rng.gauss(lam, math.sqrt(lam)))))
    l, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= l:
            return k
        k += 1


def binomial(rng, n, p):
    return sum(1 for _ in range(n) if rng.random() < p)


def weighted_choice(rng, weights):
    keys = list(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys])[0]


def clip(x, lo, hi):
    return max(lo, min(hi, x))


def round_price(p):
    if p < 50:
        return max(5, int(round(p / 5.0)) * 5)
    return int(round(p / 10.0)) * 10


# --------------------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------------------

def build_locations(rng, areas_per_city):
    locations, loc_ing = [], []
    for city, (state, tier, cost_idx, enforce, city_shift) in CITIES.items():
        # each city has 1-3 "hotspot" ingredients with an active racket
        hotspots = set(rng.sample(list(INGREDIENTS), rng.randint(1, 3)))
        for area in rng.sample(list(AREA_TYPES), areas_per_city):
            area_shift, density = AREA_TYPES[area]
            loc_id = f"L{len(locations) + 1:03d}"
            loc_enforce = clip(enforce * rng.uniform(0.7, 1.3), 0.3, 2.0)
            locations.append({
                "location_id": loc_id, "city": city, "state": state, "city_tier": tier,
                "area_type": area, "cost_of_living_index": round(cost_idx * rng.uniform(0.93, 1.07), 3),
                "enforcement_intensity": round(loc_enforce, 3),
                "avg_income_index": round(clip(cost_idx * rng.uniform(0.8, 1.2) - area_shift * 0.3, 0.4, 2.0), 3),
                "population_density_k_per_km2": round(rng.uniform(3, 30) * density, 1),
                "_density": density, "_hotspots": hotspots, "_cost": cost_idx,
            })
            for ing, (base_p, mkt_price, cs, _, _) in INGREDIENTS.items():
                hot = ing in hotspots
                true_logit = (logit(base_p) + city_shift + area_shift
                              + (rng.uniform(0.9, 1.5) if hot else 0.0) + rng.gauss(0, 0.3))
                p_true = sigmoid(true_logit)
                tested = poisson(rng, 18 * loc_enforce * (1.4 if hot else 1.0))
                failed = binomial(rng, tested, clip(p_true * 0.9, 0, 1))  # 90% detection
                mkt = mkt_price * cost_idx * rng.uniform(0.92, 1.08)
                loc_ing.append({
                    "location_id": loc_id, "ingredient": ing,
                    "samples_tested_12m": tested, "samples_failed_12m": failed,
                    "sample_fail_rate_12m": round(failed / tested, 4) if tested else "",
                    "seizures_12m": poisson(rng, p_true * loc_enforce * 6),
                    "news_alerts_12m": poisson(rng, p_true * 3 * (1.6 if hot else 1.0)),
                    "market_price_per_kg": round(mkt, 2),
                    # cheap (fake) stock in circulation drags the lowest wholesale quote down
                    "min_wholesale_price_per_kg": round(mkt * (1 - 0.6 * p_true) * rng.uniform(0.92, 1.0), 2),
                    "is_known_hotspot": int(hot and rng.random() < 0.7),  # not every racket is public
                    "_p_true": p_true,
                })
    return locations, loc_ing


def build_restaurants(rng, n, locations):
    type_w = {t: v[4] for t, v in RESTAURANT_TYPES.items()}
    loc_w = [l["_density"] * rng.uniform(0.6, 1.4) for l in locations]
    restaurants = []
    for i in range(n):
        loc = rng.choices(locations, weights=loc_w)[0]
        rtype = weighted_choice(rng, type_w)
        price_mult, integ_bias, (mn, mx), allowed, _ = RESTAURANT_TYPES[rtype]
        # upscale areas skew away from street stalls / dhabas
        if loc["area_type"] in ("upscale_commercial", "it_corridor") and rtype in ("street_stall", "dhaba") and rng.random() < 0.4:
            rtype = rng.choice(["qsr", "casual_dining", "cloud_kitchen"])
            price_mult, integ_bias, (mn, mx), allowed, _ = RESTAURANT_TYPES[rtype]

        sup_w = {k: v[1] for k, v in SUPPLIERS.items()}
        if rtype in ("fine_dining", "hotel_restaurant", "qsr"):
            sup_w["branded_distributor"] *= 2.5
        if rtype in ("street_stall", "dhaba"):
            sup_w["wholesale_mandi"] *= 1.8
            sup_w["local_vendor"] *= 1.5
        supplier = weighted_choice(rng, sup_w)

        z = rng.gauss(integ_bias, 0.9)  # latent integrity (hidden)
        price_pressure = rng.gauss(0.5 - price_mult * 0.35, 0.5)  # hidden: competition / margin squeeze

        lic_score = 0.8 * z + {"street_stall": -1.2, "dhaba": -0.5, "cloud_kitchen": -0.2}.get(rtype, 0.4) + rng.gauss(0, 0.7)
        if lic_score < -1.0:
            licence = "none"
        elif lic_score < 0.3:
            licence = "fssai_registration"
        elif lic_score < 1.4 or rtype not in ("hotel_restaurant", "fine_dining"):
            licence = "fssai_state_licence"
        else:
            licence = "fssai_central_licence"

        hygiene = "" if rng.random() < (0.55 if rtype == "street_stall" else 0.25) else int(round(clip(3 + 0.9 * z + rng.gauss(0, 0.8), 0, 5)))
        years = round(clip(rng.expovariate(1 / 7.0), 0.2, 45), 1)
        reviews = int(rng.lognormvariate(4.5 + 0.3 * price_mult, 1.0))
        on_agg = rng.random() < {"street_stall": 0.15, "dhaba": 0.35, "cloud_kitchen": 1.0, "fine_dining": 0.4}.get(rtype, 0.7)

        restaurants.append({
            "restaurant_id": f"R{i + 1:05d}",
            "restaurant_name": f"{rng.choice(NAME_A)} {rng.choice(NAME_B)}",
            "location_id": loc["location_id"],
            "restaurant_type": rtype,
            "primary_cuisine": CUISINES[rng.choice(allowed)],
            "fssai_licence": licence,
            "hygiene_rating": hygiene,
            "years_operating": years,
            "seating_capacity": 0 if rtype in ("cloud_kitchen", "street_stall") else int(rng.uniform(15, 40) * price_mult * rng.uniform(0.8, 2.5)),
            "online_rating": round(clip(3.9 + 0.2 * z + rng.gauss(0, 0.35), 1.0, 5.0), 1),
            "num_reviews": reviews,
            "complaints_12m": poisson(rng, math.exp(-0.6 - 0.5 * z) * (1 + reviews / 400)),
            "prior_violations_3y": poisson(rng, math.exp(-0.7 - 0.7 * z)),
            "months_since_last_inspection": int(clip(rng.expovariate(1 / (30 if licence == "none" else 14)), 1, 72)),
            "supplier_type": supplier,
            "on_delivery_apps": int(on_agg),
            "avg_app_discount_pct": round(clip(rng.gauss(18 - 6 * z + 10 * price_pressure, 8), 0, 60), 1) if on_agg else "",
            "_z": z, "_pp": price_pressure, "_price_mult": price_mult, "_allowed": allowed,
            "_menu_range": (mn, mx), "_loc": loc,
        })
    return restaurants


def build_menus(rng, restaurants, loc_ing_index):
    dishes_by_cuisine = defaultdict(list)
    for d in DISHES:
        for c in d[6]:
            dishes_by_cuisine[c].append(d)

    items = []
    for r in restaurants:
        loc = r["_loc"]
        primary_code = next(k for k, v in CUISINES.items() if v == r["primary_cuisine"])
        pool_primary = dishes_by_cuisine[primary_code]
        pool_other = [d for c in r["_allowed"] for d in dishes_by_cuisine[c]]
        n_items = rng.randint(*r["_menu_range"])
        chosen = {}
        attempts = 0
        while len(chosen) < n_items and attempts < n_items * 6:
            attempts += 1
            d = rng.choice(pool_primary if rng.random() < 0.7 else pool_other)
            chosen[d[0]] = d

        sup_eff = SUPPLIERS[r["supplier_type"]][0]
        lic_eff = {"none": 0.45, "fssai_registration": 0.15}.get(r["fssai_licence"], 0.0)
        for dish, ing, grams, base_price, cat, veg, _ in chosen.values():
            li = loc_ing_index[(loc["location_id"], ing)]
            _, _, cs, detect, adulterants = INGREDIENTS[ing]
            fair_price = base_price * r["_price_mult"] * loc["_cost"] * rng.lognormvariate(0, 0.12)
            ing_cost = grams / 1000.0 * li["market_price_per_kg"]
            cost_share = clip(ing_cost / fair_price, 0, 1.5)

            s_eff = sup_eff - (0.3 if r["supplier_type"] == "own_procurement" and ing in DAIRY else 0)
            z_logit = (logit(li["_p_true"]) - 1.4 * r["_z"] + 0.8 * cs * r["_pp"]
                       + 1.3 * (cost_share - 0.25) + s_eff + lic_eff + INTERCEPT)
            adulterated = rng.random() < sigmoid(z_logit)

            price = fair_price
            if adulterated and rng.random() < 0.75:
                # cost savings passed on to undercut competitors
                price *= clip(1 - rng.uniform(0.15, 0.45) * cs, 0.45, 0.95)
            elif not adulterated and rng.random() < 0.07:
                price *= rng.uniform(0.70, 0.90)  # honest promos / loss leaders (false-positive noise)
            price = round_price(price)

            review_mentions = poisson(rng, (0.15 + (1.8 * detect if adulterated else 0)) * (0.3 + min(r["num_reviews"], 2000) / 600))
            items.append({
                "item_id": f"I{len(items) + 1:06d}",
                "restaurant_id": r["restaurant_id"],
                "dish_name": dish,
                "dish_category": cat,
                "veg_nonveg": veg,
                "primary_ingredient": ing,
                "ingredient_grams_est": grams,
                "menu_price_inr": price,
                "est_ingredient_cost_inr": round(ing_cost, 2),
                "quality_complaint_mentions": review_mentions,
                "is_adulterated": int(adulterated),
                "adulterant_detected": rng.choice(adulterants) if adulterated else "none",
                "_rtype": r["restaurant_type"], "_city": loc["city"], "_loc_id": loc["location_id"],
                "_cost": loc["_cost"],
            })
    return items


def add_price_features(items):
    """Compare each item's price to peers — the 'unusually cheap paneer' signal."""
    peer = defaultdict(list)       # same city, same restaurant type, same dish
    natl = defaultdict(list)       # nationwide, same type, same dish (normalised by city cost)
    local = defaultdict(list)      # same location, same dish, any type
    for it in items:
        p = it["menu_price_inr"]
        peer[(it["_city"], it["_rtype"], it["dish_name"])].append(p)
        natl[(it["_rtype"], it["dish_name"])].append(p / it["_cost"])
        local[(it["_loc_id"], it["dish_name"])].append(p)
    med = lambda v: statistics.median(v)
    peer_m = {k: (med(v), len(v)) for k, v in peer.items()}
    natl_m = {k: med(v) for k, v in natl.items()}
    local_m = {k: (med(v), len(v)) for k, v in local.items()}
    for it in items:
        m, n = peer_m[(it["_city"], it["_rtype"], it["dish_name"])]
        if n < 3:
            m = natl_m[(it["_rtype"], it["dish_name"])] * it["_cost"]
        lm, ln = local_m[(it["_loc_id"], it["dish_name"])]
        it["peer_group_size"] = n
        it["peer_median_price_inr"] = round(m, 1)
        it["price_ratio_to_peer_median"] = round(it["menu_price_inr"] / m, 4)
        it["local_median_price_inr"] = round(lm, 1)
        it["local_listing_count"] = ln
        it["price_ratio_to_local_median"] = round(it["menu_price_inr"] / lm, 4)
        it["price_to_ingredient_cost_ratio"] = round(it["menu_price_inr"] / max(it["est_ingredient_cost_inr"], 0.5), 3)


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def public(row):
    return [k for k in row if not k.startswith("_")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restaurants", type=int, default=8000)
    ap.add_argument("--areas-per-city", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)

    locations, loc_ing = build_locations(rng, args.areas_per_city)
    loc_ing_index = {(r["location_id"], r["ingredient"]): r for r in loc_ing}
    restaurants = build_restaurants(rng, args.restaurants, locations)
    items = build_menus(rng, restaurants, loc_ing_index)
    add_price_features(items)

    # location-level aggregates
    loc_counts = defaultdict(int)
    for r in restaurants:
        loc_counts[r["location_id"]] += 1
    for l in locations:
        rows = [loc_ing_index[(l["location_id"], ing)] for ing in INGREDIENTS]
        t = sum(x["samples_tested_12m"] for x in rows)
        l["num_restaurants"] = loc_counts[l["location_id"]]
        l["overall_sample_fail_rate_12m"] = round(sum(x["samples_failed_12m"] for x in rows) / t, 4) if t else ""
        l["total_seizures_12m"] = sum(x["seizures_12m"] for x in rows)
        l["known_hotspot_ingredients"] = ";".join(sorted(x["ingredient"] for x in rows if x["is_known_hotspot"]))

    # restaurant-level aggregates + labels
    by_rest = defaultdict(list)
    for it in items:
        by_rest[it["restaurant_id"]].append(it)
    high_risk = {"paneer", "khoya", "ghee", "soya", "milk", "saffron", "honey"}
    for r in restaurants:
        its = by_rest[r["restaurant_id"]]
        loc_id = r["location_id"]
        n_adult = sum(i["is_adulterated"] for i in its)
        r["menu_size"] = len(its)
        r["avg_menu_price_inr"] = round(sum(i["menu_price_inr"] for i in its) / len(its), 1)
        r["share_high_risk_ingredient_items"] = round(sum(i["primary_ingredient"] in high_risk for i in its) / len(its), 4)
        r["share_items_on_local_hotspot_ingredient"] = round(sum(loc_ing_index[(loc_id, i["primary_ingredient"])]["is_known_hotspot"] for i in its) / len(its), 4)
        r["mean_price_ratio_to_peer"] = round(sum(i["price_ratio_to_peer_median"] for i in its) / len(its), 4)
        r["min_price_ratio_to_peer"] = round(min(i["price_ratio_to_peer_median"] for i in its), 4)
        r["share_items_priced_below_75pct_peer"] = round(sum(i["price_ratio_to_peer_median"] < 0.75 for i in its) / len(its), 4)
        r["total_quality_complaint_mentions"] = sum(i["quality_complaint_mentions"] for i in its)
        r["n_items_adulterated"] = n_adult
        r["share_items_adulterated"] = round(n_adult / len(its), 4)
        r["any_adulterated"] = int(n_adult > 0)
        share = n_adult / len(its)
        r["risk_level"] = "low" if n_adult == 0 else ("medium" if share < 0.25 else "high")
        u = rng.random()
        r["split"] = "train" if u < 0.70 else ("val" if u < 0.85 else "test")  # grouped by restaurant

    # flat item-level training table
    rest_idx = {r["restaurant_id"]: r for r in restaurants}
    loc_idx = {l["location_id"]: l for l in locations}
    rest_cols = ["location_id", "restaurant_type", "primary_cuisine", "fssai_licence", "hygiene_rating",
                 "years_operating", "seating_capacity", "online_rating", "num_reviews", "complaints_12m",
                 "prior_violations_3y", "months_since_last_inspection", "supplier_type", "on_delivery_apps",
                 "avg_app_discount_pct", "menu_size", "avg_menu_price_inr", "share_high_risk_ingredient_items",
                 "share_items_on_local_hotspot_ingredient", "mean_price_ratio_to_peer"]
    loc_cols = ["city", "state", "city_tier", "area_type", "cost_of_living_index", "enforcement_intensity",
                "avg_income_index", "population_density_k_per_km2", "overall_sample_fail_rate_12m"]
    li_cols = ["samples_tested_12m", "sample_fail_rate_12m", "seizures_12m", "news_alerts_12m",
               "market_price_per_kg", "min_wholesale_price_per_kg", "is_known_hotspot"]
    item_cols = ["item_id", "restaurant_id", "dish_name", "dish_category", "veg_nonveg", "primary_ingredient",
                 "ingredient_grams_est", "menu_price_inr", "est_ingredient_cost_inr", "peer_group_size",
                 "peer_median_price_inr", "price_ratio_to_peer_median", "local_median_price_inr",
                 "local_listing_count", "price_ratio_to_local_median", "price_to_ingredient_cost_ratio",
                 "quality_complaint_mentions"]
    flat = []
    for it in items:
        r = rest_idx[it["restaurant_id"]]
        l = loc_idx[r["location_id"]]
        li = loc_ing_index[(r["location_id"], it["primary_ingredient"])]
        row = {c: it[c] for c in item_cols}
        row.update({c: r[c] for c in rest_cols})
        row.update({c: l[c] for c in loc_cols})
        row.update({f"loc_ing_{c}": li[c] for c in li_cols})
        row.update({"split": r["split"], "is_adulterated": it["is_adulterated"],
                    "adulterant_detected": it["adulterant_detected"]})
        flat.append(row)

    ing_rows = [{"ingredient": k, "national_base_prevalence": v[0], "reference_price_per_kg": v[1],
                 "known_adulterants": ";".join(v[4])} for k, v in INGREDIENTS.items()]
    dish_rows = [{"dish_name": d[0], "primary_ingredient": d[1], "ingredient_grams_est": d[2],
                  "reference_price_inr": d[3], "dish_category": d[4], "veg_nonveg": d[5],
                  "cuisines": ";".join(CUISINES[c] for c in d[6])} for d in DISHES]

    loc_fields = ["location_id", "city", "state", "city_tier", "area_type", "cost_of_living_index",
                  "enforcement_intensity", "avg_income_index", "population_density_k_per_km2",
                  "num_restaurants", "overall_sample_fail_rate_12m", "total_seizures_12m", "known_hotspot_ingredients"]
    write_csv(os.path.join(args.out, "locations.csv"), locations, loc_fields)
    write_csv(os.path.join(args.out, "location_ingredient_stats.csv"), loc_ing, public(loc_ing[0]))
    write_csv(os.path.join(args.out, "ingredients.csv"), ing_rows, public(ing_rows[0]))
    write_csv(os.path.join(args.out, "dishes.csv"), dish_rows, public(dish_rows[0]))
    write_csv(os.path.join(args.out, "restaurants.csv"), restaurants, public(restaurants[0]))
    write_csv(os.path.join(args.out, "menu_items.csv"), items, public(items[0]))
    write_csv(os.path.join(args.out, "training_items_flat.csv"), flat, list(flat[0]))

    pos = sum(i["is_adulterated"] for i in items)
    print(f"locations={len(locations)} restaurants={len(restaurants)} items={len(items)}")
    print(f"item positive rate={pos / len(items):.3f}  "
          f"restaurants with any adulteration={sum(r['any_adulterated'] for r in restaurants) / len(restaurants):.3f}")
    for lvl in ("low", "medium", "high"):
        print(f"  risk_level={lvl}: {sum(r['risk_level'] == lvl for r in restaurants)}")


if __name__ == "__main__":
    main()