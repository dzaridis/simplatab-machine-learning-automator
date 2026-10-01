"""Synthetic example for the Time Series Forecasting automator: daily mean blood glucose of
patients with type 1 diabetes. Writes Train.csv and Test.csv next to this script.

Columns (long format, one row per patient and day):
- ID: patient; Time: date; Target: daily mean glucose (mg/dL);
- Insulin_units: prescribed daily insulin, known in advance (a future covariate);
- Weekend: 1 on Saturdays and Sundays, known in advance (a future covariate);
- Steps: daily step count from a wearable, only known up to today (a past covariate);
- Age, Sex, BMI: static features.

Test.csv holds 14-day horizons: the next 14 days of 30 training patients (temporal hold-out)
and 10 new patients with 60 days of history followed by the 14 days to forecast.
"""
import os

import numpy as np
import pandas as pd

HORIZON = 14


def patient(rng, pid, start, days):
    age = int(rng.integers(18, 75))
    sex = rng.choice(["F", "M"])
    bmi = round(float(rng.normal(26, 4)), 1)
    dates = pd.date_range(start, periods=days, freq="D")
    weekend = (dates.dayofweek >= 5).astype(int)
    # Insulin prescription: a base dose adjusted every 3-4 weeks
    base = rng.normal(40, 8) + 0.4 * (bmi - 26)
    changes = np.cumsum(rng.choice([0, 0, 0, 2, -2, 4], size=days) * (rng.random(days) < 0.04))
    insulin = np.round(np.clip(base + changes, 15, 90)).astype(int)
    steps = np.clip(rng.normal(7000 - 60 * (age - 40), 1800, days) - 1500 * weekend, 500, None).round(-1).astype(int)
    level = 120 + 0.6 * (age - 40) + 1.8 * (bmi - 26) + (6 if sex == "M" else 0)
    glucose = np.empty(days)
    noise = 0.0
    for t in range(days):
        noise = 0.6 * noise + rng.normal(0, 8)
        activity = -0.002 * (steps[t - 1] - 7000) if t else 0.0  # yesterday's activity lowers glucose
        glucose[t] = level + 14 * weekend[t] - 1.1 * (insulin[t] - base) + activity + 5 * np.sin(2 * np.pi * t / 30) + noise
    return pd.DataFrame({"ID": pid, "Time": dates.strftime("%Y-%m-%d"), "Target": glucose.round(1),
                         "Insulin_units": insulin, "Weekend": weekend, "Steps": steps,
                         "Age": age, "Sex": sex, "BMI": bmi})


def main(folder=os.path.dirname(os.path.abspath(__file__))):
    rng = np.random.default_rng(7)
    train, test = [], []
    for i in range(30):  # followed patients: training history, then 14 test days
        start = pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(0, 45)))
        series = patient(rng, f"P{i + 1:03d}", start, int(rng.integers(100, 160)) + HORIZON)
        train.append(series.iloc[:-HORIZON])
        test.append(series.iloc[-HORIZON:])
    for i in range(30, 40):  # new patients: 60 days of history and the 14 days to forecast
        start = pd.Timestamp("2024-03-01") + pd.Timedelta(days=int(rng.integers(0, 30)))
        test.append(patient(rng, f"P{i + 1:03d}", start, 60 + HORIZON))
    pd.concat(train).to_csv(os.path.join(folder, "Train.csv"), index=False)
    pd.concat(test).to_csv(os.path.join(folder, "Test.csv"), index=False)


if __name__ == "__main__":
    main()
