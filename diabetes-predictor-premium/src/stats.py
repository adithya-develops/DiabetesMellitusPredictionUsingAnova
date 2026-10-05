import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, norm

def conditional_probability(data, condition, target="diabetes"):
    subset = data.query(condition)
    return float(subset[target].mean()) if len(subset) else np.nan

def proportion_ci(successes, n, confidence=0.95):
    p = successes / n
    z = norm.ppf(1 - (1-confidence)/2)
    se = np.sqrt(p*(1-p)/n)
    return p, max(0.0, p-z*se), min(1.0, p+z*se)

def chi_square(data, feature, target="diabetes"):
    table = pd.crosstab(data[feature], data[target])
    chi2, p, dof, expected = chi2_contingency(table)
    return table, chi2, p, dof
