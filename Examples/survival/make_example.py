"""Synthetic example for the Survival Analysis automator: overall survival after surgery for
colorectal cancer. Writes Train.csv (800 patients) and Test.csv (300 patients of another
period) next to this script.

Columns (one row per patient):
- ID; Time: months from surgery to death or last follow-up; Event: 1 death, 0 censored;
- Age, Sex, Stage (I-IV), Grade (1-3), Tumour_size_mm, CEA (ng/mL), ECOG (0-2), Positive_nodes,
  Adjuvant_chemo (Yes/No), BMI (no effect on survival).

The hazard is not proportional for every feature: age has a U-shaped effect, chemotherapy helps
stage III patients only, and the effect of CEA levels off; follow-up ends 2 to 8 years after
surgery (administrative censoring) and 10% of the patients are lost to follow-up.
"""
import os

import numpy as np
import pandas as pd

STAGES = ["I", "II", "III", "IV"]


def cohort(rng, count, start):
    stage = rng.choice(STAGES, count, p=[0.2, 0.32, 0.33, 0.15])
    s = np.array([STAGES.index(v) for v in stage])
    age = np.clip(rng.normal(66, 11, count), 30, 92).round()
    grade = np.clip(1 + (rng.random(count) < 0.55) + (rng.random(count) < 0.2 + 0.1 * s), 1, 3).astype(int)
    size = np.clip(rng.lognormal(3.4 + 0.12 * s, 0.35, count), 5, 150).round()
    cea = np.round(rng.lognormal(1.0 + 0.45 * s, 0.8, count), 1)
    ecog = np.clip(rng.poisson(0.35 + 0.012 * np.maximum(age - 60, 0), count), 0, 2)
    nodes = np.where(s >= 2, rng.poisson(2.5 + 2 * (s == 3), count) + 1, 0)
    chemo = np.where((s >= 2) & (rng.random(count) < 0.65), "Yes", "No")
    sex = rng.choice(["F", "M"], count, p=[0.45, 0.55])
    bmi = np.round(rng.normal(27, 4.5, count), 1)
    log_hazard = (0.75 * s + 0.0012 * (age - 62) ** 2 + 0.25 * (grade - 1) + 0.35 * np.log1p(cea) / np.log(10)
                  + 0.45 * ecog + 0.06 * np.minimum(nodes, 10) + 0.004 * (size - 30)
                  - 0.6 * ((chemo == "Yes") & (s == 2)) + 0.1 * (sex == "M"))
    scale, shape = 75.0, 1.25             # Weibull baseline (months)
    u = rng.random(count)
    time = scale * (-np.log(u) / np.exp(log_hazard - log_hazard.mean())) ** (1 / shape)
    follow_up = rng.uniform(24, 96, count)                        # administrative censoring
    lost = np.where(rng.random(count) < 0.1, rng.uniform(1, 96, count), np.inf)
    end = np.minimum(follow_up, lost)
    event = (time <= end).astype(int)
    observed = np.round(np.maximum(np.minimum(time, end), 0.1), 1)
    return pd.DataFrame({"ID": [f"CRC{start + i:04d}" for i in range(count)], "Time": observed, "Event": event,
                         "Age": age.astype(int), "Sex": sex, "Stage": stage, "Grade": grade,
                         "Tumour_size_mm": size.astype(int), "CEA": cea, "ECOG": ecog, "Positive_nodes": nodes,
                         "Adjuvant_chemo": chemo, "BMI": bmi})


def main():
    rng = np.random.default_rng(2024)
    here = os.path.dirname(os.path.abspath(__file__))
    cohort(rng, 800, 1).to_csv(os.path.join(here, "Train.csv"), index=False)
    cohort(rng, 300, 801).to_csv(os.path.join(here, "Test.csv"), index=False)


if __name__ == "__main__":
    main()
