# NLP benchmark

93 training dishes, 4743 test strings (10 variants per dish per perturbation, seed 42)

### 1. Robustness: share of menu strings where all four ingredient columns are correct

| perturbation             |   exact lookup |   normalised lookup |   fuzzy match (WRatio>=85) |   NLP layer |
|:-------------------------|---------------:|--------------------:|---------------------------:|------------:|
| clean                    |          1.000 |               1.000 |                      1.000 |       1.000 |
| case                     |          0.000 |               1.000 |                      1.000 |       1.000 |
| filler                   |          0.000 |               0.000 |                      0.990 |       1.000 |
| spelling                 |          0.398 |               0.398 |                      0.829 |       0.989 |
| typo                     |          0.141 |               0.141 |                      0.970 |       0.976 |
| mixed (2-3 of the above) |          0.001 |               0.073 |                      0.756 |       0.967 |

### 1b. Robustness, all perturbations pooled

| resolver                 |   all 4 columns |   primary_ingredient |   veg_nonveg |
|:-------------------------|----------------:|---------------------:|-------------:|
| exact lookup             |           0.125 |                0.125 |        0.125 |
| normalised lookup        |           0.336 |                0.336 |        0.336 |
| fuzzy match (WRatio>=85) |           0.911 |                0.927 |        0.943 |
| NLP layer                |           0.987 |                0.991 |        0.992 |

NLP misses: 63. Examples: 'Khir' (was Kheer); 'Khir' (was Kheer); 'Khir' (was Kheer); 'Khir' (was Kheer); 'Khir' (was Kheer); 'Khir' (was Kheer); 'Khir' (was Kheer); 'Khir' (was Kheer)

### 2. Hidden ingredients (41 hand-labelled dishes)

| resolver                       |   resolved |   veg/nonveg acc |   risky-ingredient recall |   risky-ingredient precision |   all risky found |
|:-------------------------------|-----------:|-----------------:|--------------------------:|-----------------------------:|------------------:|
| NLP layer                      |      0.878 |            0.878 |                     0.859 |                        0.965 |             0.805 |
| keywords only                  |      0.512 |            0.512 |                     0.312 |                        0.952 |             0.268 |
| fuzzy match to training dishes |      0.317 |            0.293 |                     0.156 |                        0.769 |             0.049 |

### 2b. NLP layer by resolution tier

| source     |   ('veg_ok', 'mean') |   ('veg_ok', 'size') |   ('all_found', 'mean') |   ('all_found', 'size') |
|:-----------|---------------------:|---------------------:|------------------------:|------------------------:|
| kb         |                1.000 |               17.000 |                   1.000 |                  17.000 |
| kb_partial |                1.000 |               11.000 |                   1.000 |                  11.000 |
| keywords   |                1.000 |                8.000 |                   0.625 |                   8.000 |
| unknown    |                0.000 |                5.000 |                   0.000 |                   5.000 |

NLP layer missed some risky ingredient for: Mutton Galouti, Chicken Kolhapuri, Paneer 65, Veg Kolhapuri, Undhiyu, Kosha Mangsho, Bhapa Ilish, Litti Chokha

Rebuild check (true ingredient -> rebuilt features): max |dp| 0.0152, 0.01% of items differ (cent-level rounding of ingredient cost)

### 3. Downstream: 19022 test items, dish names corrupted with 2-3 perturbations each

| ingredient columns from                 |   ROC-AUC |   PR-AUC |   flag rate |   primary_ingredient correct |
|:----------------------------------------|----------:|---------:|------------:|-----------------------------:|
| original columns (oracle)               |     0.902 |    0.762 |       0.198 |                        1.000 |
| no ingredient info at all               |     0.883 |    0.712 |       0.219 |                        0.000 |
| messy names -> exact lookup             |     0.883 |    0.712 |       0.219 |                        0.003 |
| messy names -> normalised lookup        |     0.884 |    0.714 |       0.218 |                        0.050 |
| messy names -> fuzzy match (WRatio>=85) |     0.898 |    0.751 |       0.204 |                        0.816 |
| messy names -> NLP layer                |     0.902 |    0.760 |       0.197 |                        0.965 |

### 4. Throughput

- 7,375 unique messy names/sec (0.14 ms each, 1860 names)
- enrich() on all 126,851 dataset rows: 0.50s (names are resolved once per unique string)
