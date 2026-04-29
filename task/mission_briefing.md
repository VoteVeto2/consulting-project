# Mission Briefing: Insurance Customer Profiling

## Who is asking and what do they want?

**Higher management** at an insurance company wants to **identify unprofitable customers** in their portfolio so they can **take appropriate action at renewal** (e.g., reprice, adjust coverage, or drop the policy).

This is a **Statistical Consulting exam assignment** set by Robin Van Oirbeek (Senior Data Scientist at Ageas Re, KU Leuven alumnus). It simulates a real consulting engagement.

---

## What we are given: Two datasets, one portfolio

| Dataset | Records | Target variable | What it measures |
|---|---|---|---|
| `frequency.csv` | 24,774 policies | `claimNumbMD` (binary 0/1) | Did this policy have at least one Material Damage accident? |
| `severity.csv` | 12,256 policies | `claimSizeMD` (continuous, EUR) | How much did the claim cost? |

Both datasets share the **same 10 covariates** (underwriting year, gender, car type, car size, occupation, age, policy tenure, car value, coverage type, population density) but **cannot be linked at the individual policyholder level**. This is the central constraint.

The severity dataset is smaller because it only contains policies that *had* a claim — you can only measure claim cost when a claim exists. The frequency dataset is the full portfolio: roughly 50/50 split between claim (12,277) and no-claim (12,497).

This is a classic **frequency-severity decomposition** in insurance:

> **Expected Loss per customer** = $\mathbb{P}(\text{claim}) \times \mathbb{E}(\text{claim cost} | \text{claim occurred})$

We need to model each piece separately, then combine.

---

## Shared Covariates (10 variables)

| Variable | Type | Description |
|---|---|---|
| `uwYear` | Factor (2009, 2010) | Underwriting year |
| `gender` | Factor (Female, Male) | Driver gender |
| `carType` | Factor (A, B, C, D, E) | Car type |
| `carCat` | Factor (Small, Medium, Large) | Car size category |
| `job` | Factor (5 levels) | Occupation: Employed, Housewife, Retired, Self-employed, Unemployed |
| `age` | Continuous | Driver age in years |
| `nYears` | Continuous | Years the policy had been active at start |
| `carVal` | Continuous | Market value of car (EUR) |
| `cover` | Binary (0/1) | Whether material damage cover is included |
| `density` | Continuous | Population density (inhabitants/km²) of driver's region |

---

## What we are NOT given

- **No premium data.** We know about claims but not what customers are paying. "Unprofitable" requires comparing loss to revenue, so we need to reason about what "unprofitable" means without explicit premium information.
- **No individual-level linkage** between the two datasets.
- **No prescribed method.** No single correct answer.

---

## How we will be evaluated

Not on fancy models. On three things:

1. Did we **ask the right questions**? (e.g., what does "unprofitable" even mean here?)
2. Did we make **defensible choices**? (methodology, assumptions, trade-offs)
3. Can we **explain our reasoning clearly** to a non-technical audience?

The lecture context hammered this home: *"A brilliant model nobody uses is worse than a decent model everyone trusts."*

---

## Key questions we should be thinking about (but not solving yet)

- How do we define "unprofitable" without premium data?
- What does "customer profile" mean — segments? individual risk scores? covariate-driven risk factors?
- How do we combine frequency and severity models when we can't link records?
- What "appropriate action at renewal" looks like — and how that shapes what we deliver
- Which covariates matter most, and how do we communicate that to management?
