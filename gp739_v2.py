# https://github.com/guyko81/gplearn



"""
Genetic Programming
gp739_v2 
Loto 7/39 predikcija preko sklearn ensemble 
Samostalan ensemble fajl 
"""


import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.neighbors import KNeighborsRegressor


CSV_PATH = "/Users/4c/Desktop/GHQ/data/loto7hh_4592_k27.csv"
COLS = ["Num1", "Num2", "Num3", "Num4", "Num5", "Num6", "Num7"]
SEED = 39
np.random.seed(SEED)

MIN_POS = np.array([1, 2, 3, 4, 5, 6, 7], dtype=int)
MAX_POS = np.array([33, 34, 35, 36, 37, 38, 39], dtype=int)
FEATURE_COLS = [f"f{i+1}" for i in range(7)]


def load_draws(path):
    df = pd.read_csv(path)
    if all(c in df.columns for c in COLS):
        return df[COLS].values.astype(float)
    return pd.read_csv(path, header=None).iloc[:, :7].values.astype(float)


def enforce_loto_7_39(nums):
    nums = np.rint(np.asarray(nums, dtype=float)).astype(int)
    nums = np.clip(nums, MIN_POS, MAX_POS)
    nums = np.sort(nums)

    for i in range(7):
        low = MIN_POS[i] if i == 0 else max(MIN_POS[i], nums[i - 1] + 1)
        nums[i] = max(nums[i], low)

    for i in range(6, -1, -1):
        high = MAX_POS[i] if i == 6 else min(MAX_POS[i], nums[i + 1] - 1)
        nums[i] = min(nums[i], high)

    for i in range(7):
        low = MIN_POS[i] if i == 0 else max(MIN_POS[i], nums[i - 1] + 1)
        nums[i] = max(nums[i], low)

    return nums


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def make_builders(seed):
    return [
        ("rf", lambda: RandomForestRegressor(
            n_estimators=500, max_depth=16, min_samples_leaf=2, random_state=seed, n_jobs=-1
        )),
        ("etr", lambda: ExtraTreesRegressor(
            n_estimators=500, max_depth=18, min_samples_leaf=2, random_state=seed, n_jobs=-1
        )),
        ("gbr", lambda: GradientBoostingRegressor(
            n_estimators=450, learning_rate=0.04, max_depth=3, random_state=seed
        )),
        ("ada", lambda: AdaBoostRegressor(
            n_estimators=400, learning_rate=0.05, random_state=seed
        )),
        ("hgb", lambda: HistGradientBoostingRegressor(
            max_depth=6, learning_rate=0.04, max_iter=500, random_state=seed
        )),
        ("knn", lambda: KNeighborsRegressor(
            n_neighbors=45, weights="distance", p=2
        )),
    ]


def main():
    draws = load_draws(CSV_PATH)
    X = pd.DataFrame(draws[:-1], columns=FEATURE_COLS)
    Y = draws[1:]
    X_next = pd.DataFrame(draws[-1:].astype(float), columns=FEATURE_COLS)

    val_size = min(450, max(200, len(X) // 10))
    X_train, X_val = X.iloc[:-val_size], X.iloc[-val_size:]

    preds = []
    print("=" * 72)
    print("gp739_v2 - Standalone ensemble Lotto 7/39")
    print("=" * 72)
    print(f"CSV: {CSV_PATH}")
    print(f"Uzoraka za trening: {len(X)}")
    print()

    for pos in range(7):
        y = Y[:, pos]
        y_train, y_val = y[:-val_size], y[-val_size:]
        candidates = []
        print(f"[pozicija {pos + 1}]")

        for rep in range(2):
            builders = make_builders(SEED + pos * 20 + rep)
            for name, build in builders:
                model = build()
                model.fit(X_train, y_train)
                val_pred = model.predict(X_val)
                score = mae(y_val, val_pred)
                next_pred = float(np.asarray(model.predict(X_next)).ravel()[0])
                candidates.append((score, next_pred))
                print(f"  {name}/r{rep + 1}: mae={score:.4f}, next={next_pred:.6f}")

        candidates.sort(key=lambda x: x[0])
        best_next = [candidates[0][1], candidates[1][1], candidates[2][1]]
        p = float(np.median(best_next))
        preds.append(p)
        print(f"  => best-3 median: {p:.6f}")

    final_pred = enforce_loto_7_39(preds)
    print()
    print("Raw ensemble (pre enforce):", np.round(np.asarray(preds, dtype=float), 6))
    print("Predicted next loto 7/39 combination:", final_pred)
    print("=" * 72)


if __name__ == "__main__":
    main()



################################################################################
################################################################################



"""
v2 ima drugu logiku (sklearn ensemble) 

po poziciji ima 6 različitih modela x 2 rep = 12 kandidata 

više prolaza za bolju stabilnost: 
za svaku poziciju 1..7
radi rep=2
i u svakom rep-u trenira 6 modela (rf, etr, gbr, ada, hgb, knn)
Ukupno: 7 x 2 x 6 = 84 treninga za svaku poziciju 
"""



"""
========================================================================
gp739_v2 - Standalone ensemble Lotto 7/39
========================================================================
CSV: /Users/4c/Desktop/GHQ/data/loto7hh_4592_k27.csv
Uzoraka za trening: 4591

[pozicija 1]
  rf/r1: mae=3.0852, next=4.785486
  etr/r1: mae=3.1010, next=4.663989
  gbr/r1: mae=3.0916, next=x
  ada/r1: mae=3.4218, next=y
  hgb/r1: mae=3.1200, next=4.523210
  knn/r1: mae=3.0852, next=4.853921
  rf/r2: mae=3.0877, next=4.885320
  etr/r2: mae=3.1097, next=4.580544
  gbr/r2: mae=3.0884, next=y
  ada/r2: mae=3.3651, next=y
  hgb/r2: mae=3.1200, next=4.523210
  knn/r2: mae=3.0852, next=4.853921
  => best-3 median: 4.853921
[pozicija 2]
  rf/r1: mae=4.4257, next=9.659237
  etr/r1: mae=4.4253, next=9.719988
  gbr/r1: mae=4.4239, next=x
  ada/r1: mae=4.6101, next=x
  hgb/r1: mae=4.5252, next=9.309587
  knn/r1: mae=4.4006, next=10.540226
  rf/r2: mae=4.4164, next=9.449305
  etr/r2: mae=4.4322, next=10.178508
  gbr/r2: mae=4.4230, next=y
  ada/r2: mae=4.6145, next=y
  hgb/r2: mae=4.5252, next=9.309587
  knn/r2: mae=4.4006, next=10.540226
  => best-3 median: x
[pozicija 3]
  rf/r1: mae=5.1357, next=15.340625
  etr/r1: mae=5.1183, next=15.694765
  gbr/r1: mae=5.0907, next=x
  ada/r1: mae=5.0066, next=x
  hgb/r1: mae=5.2299, next=15.344764
  knn/r1: mae=5.0136, next=16.814997
  rf/r2: mae=5.1286, next=15.489410
  etr/r2: mae=5.1255, next=15.891788
  gbr/r2: mae=5.0915, next=y
  ada/r2: mae=5.0126, next=y
  hgb/r2: mae=5.2299, next=15.344764
  knn/r2: mae=5.0136, next=16.814997
  => best-3 median: y
[pozicija 4]
  rf/r1: mae=5.4246, next=20.010940
  etr/r1: mae=5.4373, next=20.583189
  gbr/r1: mae=5.3547, next=x
  ada/r1: mae=5.2559, next=x
  hgb/r1: mae=5.5100, next=19.686323
  knn/r1: mae=5.2914, next=21.340221
  rf/r2: mae=5.4173, next=19.769629
  etr/r2: mae=5.4245, next=20.355056
  gbr/r2: mae=5.3546, next=y
  ada/r2: mae=5.2532, next=y
  hgb/r2: mae=5.5100, next=19.686323
  knn/r2: mae=5.2914, next=21.340221
  => best-3 median: 20.161603
[pozicija 5]
  rf/r1: mae=4.8223, next=25.273633
  etr/r1: mae=4.8514, next=25.148051
  gbr/r1: mae=4.7156, next=x
  ada/r1: mae=4.7923, next=x
  hgb/r1: mae=4.9116, next=23.684319
  knn/r1: mae=4.7170, next=25.744931
  rf/r2: mae=4.8257, next=25.234068
  etr/r2: mae=4.8483, next=25.465788
  gbr/r2: mae=4.7141, next=y
  ada/r2: mae=4.7981, next=y
  hgb/r2: mae=4.9116, next=23.684319
  knn/r2: mae=4.7170, next=25.744931
  => best-3 median: 24.271018
[pozicija 6]
  rf/r1: mae=4.2302, next=30.180441
  etr/r1: mae=4.2636, next=30.408026
  gbr/r1: mae=4.1887, next=x
  ada/r1: mae=4.3538, next=x
  hgb/r1: mae=4.2634, next=28.501873
  knn/r1: mae=4.1900, next=30.932442
  rf/r2: mae=4.2308, next=30.139149
  etr/r2: mae=4.2502, next=30.624414
  gbr/r2: mae=4.1912, next=y
  ada/r2: mae=4.3567, next=y
  hgb/r2: mae=4.2634, next=28.501873
  knn/r2: mae=4.1900, next=30.932442
  => best-3 median: 30.932442
[pozicija 7]
  rf/r1: mae=3.0426, next=35.511518
  etr/r1: mae=3.0328, next=35.638042
  gbr/r1: mae=2.9859, next=34.692912
  ada/r1: mae=3.5050, next=x
  hgb/r1: mae=3.0740, next=x
  knn/r1: mae=2.9916, next=35.682246
  rf/r2: mae=3.0338, next=35.663393
  etr/r2: mae=3.0236, next=35.629908
  gbr/r2: mae=2.9871, next=y
  ada/r2: mae=3.4718, next=y
  hgb/r2: mae=3.0740, next=34.263896
  knn/r2: mae=2.9916, next=35.682246
  => best-3 median: 34.692912

Raw ensemble (pre enforce): 
[ 4.853921 10.540226 15.514993 20.161603 24.271018 30.932442 34.692912]
Predicted next loto 7/39 combination: 
[ 5 x y 20 24 31 35]
========================================================================
"""
