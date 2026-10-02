"""Synthetic example for the Clustering automator: adults with newly diagnosed diabetes, after the
five subgroups of Ahlqvist et al. (Lancet Diabetes Endocrinol 2018). Writes Train.csv and Test.csv
next to this script.

Columns (one row per patient):
- ID: patient;
- Age (years at diagnosis), BMI (kg/m2), HbA1c (mmol/mol), HOMA2_B (beta-cell function, %),
  HOMA2_IR (insulin resistance), GADA (glutamic acid decarboxylase antibodies: Positive/Negative),
  Sex, Systolic_BP (mmHg; does not differ between the subgroups);
- Target: the subgroup (SAID, SIDD, SIRD, MOD, MARD). Only used to evaluate the clusters: remove
  the column to cluster without labels.

HOMA2_IR has a few missing values (imputed by the automator).
"""
import os

import numpy as np
import pandas as pd

SUBGROUPS = {
    # share, Age, BMI, HbA1c, HOMA2_B, HOMA2_IR (mean, sd), P(GADA positive)
    "SAID": (0.08, (45, 12), (26, 4), (85, 14), (40, 14), (1.9, 0.5), 0.95),
    "SIDD": (0.18, (55, 10), (28, 4), (92, 13), (45, 15), (2.2, 0.6), 0.03),
    "SIRD": (0.15, (66, 9), (34, 4), (52, 8), (165, 35), (5.6, 1.1), 0.03),
    "MOD": (0.22, (46, 8), (37, 4), (55, 9), (110, 25), (3.3, 0.7), 0.03),
    "MARD": (0.37, (68, 8), (28, 3), (50, 7), (95, 20), (2.4, 0.5), 0.03),
}


def patients(rng, count, start):
    names = list(SUBGROUPS)
    shares = np.array([SUBGROUPS[s][0] for s in names])
    groups = rng.choice(names, size=count, p=shares / shares.sum())
    rows = []
    for i, group in enumerate(groups):
        _, age, bmi, hba1c, beta, ir, gada = SUBGROUPS[group]
        rows.append({
            "ID": f"P{start + i:04d}",
            "Age": int(np.clip(rng.normal(*age), 25, 90)),
            "Sex": rng.choice(["F", "M"], p=[0.4, 0.6]),
            "BMI": round(float(np.clip(rng.normal(*bmi), 17, 55)), 1),
            "HbA1c": round(float(np.clip(rng.normal(*hba1c), 38, 140)), 0),
            "HOMA2_B": round(float(np.clip(rng.normal(*beta), 5, 300)), 1),
            "HOMA2_IR": round(float(np.clip(rng.normal(*ir), 0.4, 12)), 2),
            "GADA": "Positive" if rng.random() < gada else "Negative",
            "Systolic_BP": int(rng.normal(135, 15)),
            "Target": group,
        })
    frame = pd.DataFrame(rows)
    missing = rng.random(len(frame)) < 0.02
    frame.loc[missing, "HOMA2_IR"] = np.nan
    return frame


def main():
    rng = np.random.default_rng(2018)
    here = os.path.dirname(os.path.abspath(__file__))
    patients(rng, 600, 1).to_csv(os.path.join(here, "Train.csv"), index=False)
    patients(rng, 200, 601).to_csv(os.path.join(here, "Test.csv"), index=False)


if __name__ == "__main__":
    main()
