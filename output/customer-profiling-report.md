# Identifying High-Cost Motor Insurance Customers

## A Data-Driven Framework for Renewal Decisions

---

**Prepared for:** Portfolio Management & Underwriting  
**Prepared by:** Statistical Consulting Team, KU Leuven  
**Date:** May 2026  
**Classification:** Internal

---

## Executive Summary

Your motor insurance portfolio contains a small group of customers who
generate disproportionately high claims. We built a scoring model that
identifies these customers before renewal, giving you the information
needed to take targeted action — reprice, adjust coverage, or escalate
for underwriting review.

**The headline numbers:**

| | |
|---|---|
| Customers analysed | 24,774 policies |
| Average expected claim cost | 428 EUR per policy |
| Top 5 % of customers | Expected claim cost of 1,029 EUR — **2.4 times** the portfolio average |
| Top 5 % share of total claims | ~12 % of all expected losses come from just 5 % of policies |
| Model accuracy | Predicted total claims within 0.1 % of actual total claims |

**What this means in practice:** if you take no action, a small fraction
of your book will continue to generate outsized losses. The model
identifies who they are and — critically — *why* they are expensive, so
you can choose the right intervention for each case.

**Important caveat:** we did not have access to premium data. We cannot
tell you which customers are literally loss-making today. What we can
tell you is which customers have the highest *expected claim cost*. If
your current pricing does not already account for their elevated risk,
these are the customers most likely to be unprofitable.

---

## The Challenge

You asked us to find your unprofitable customers. The data you provided
contains two files:

- **Frequency data** — 24,774 policies with information on whether each
  customer filed a material-damage claim
- **Severity data** — 12,252 claims with the cost of each claim

The data includes customer age, years of tenure, employment status,
vehicle type and value, urban density of their area, and type of cover.

**What the data does not include:** how much each customer actually pays
in premium. Without that, we cannot calculate a true profit-and-loss per
customer. Instead, we answer the next-best question: *which customers
are expected to cost you the most in claims?* These are the customers
most at risk of being underpriced.

## Our Approach — In Plain Terms

We built a model that answers two questions for every customer:

1. **How likely is this customer to file a claim?** (the "frequency"
   question)
2. **If they do claim, how expensive will it be?** (the "severity"
   question)

Multiplying these two answers gives an **expected claim cost** per
customer — the amount you should expect to pay out on average for that
policy. We then rank all 24,774 customers from cheapest to most
expensive and group them into tiers.

The model was validated in three ways:

- **Forward-looking test:** we trained on 2009 data and predicted 2010 —
  the model remained accurate, confirming it works for future renewals.
- **Independent check:** a second, completely different modelling
  technique (machine-learning) produced nearly identical customer
  rankings (88 % agreement).
- **Aggregate accuracy:** predicted total portfolio claims matched
  actual observed claims to within 0.1 %.

---

## Key Findings

### 1. Risk is concentrated in a small group

Half of your portfolio (the bottom 50 %) has an average expected claim
cost of just **249 EUR**. The most expensive 5 % averages **1,029 EUR**
— more than four times as much.

| Risk group | Share of customers | Avg expected claim cost | Share of total claims |
|---|---|---|---|
| **Very High** | 5 % | 1,029 EUR | 12 % |
| **High** | 15 % | 715 EUR | 25 % |
| **Medium** | 30 % | 483 EUR | 34 % |
| **Low** | 50 % | 249 EUR | 29 % |

The top 20 % of customers are responsible for **37 %** of your expected
claims. Targeted action on this group alone would address over a third
of your loss exposure.

### 2. There are two different kinds of expensive customer

Not all high-cost customers are expensive for the same reason. We
decomposed the portfolio into four groups based on *why* they cost
money:

| Type | What they look like | % of portfolio | % of claims |
|---|---|---|---|
| **Frequent AND costly** | Claim often, and each claim is expensive | 34 % | 54 % |
| **Frequent but cheap** | Claim often, but individual claims are small | 16 % | 15 % |
| **Rare but costly** | Rarely claim, but when they do it is expensive | 16 % | 12 % |
| **Low risk** | Rarely claim and claims are small | 34 % | 19 % |

This distinction matters because the right action is different for each
type (see Recommended Actions below).

### 3. The typical high-cost customer

Your most expensive 5 % of customers share a clear profile:

| Characteristic | Typical value |
|---|---|
| Age | ~23 years old |
| Gender | Male (85 %) |
| Employment | Unemployed (63 %) |
| Area | Urban / high-density (density score ~237) |
| Vehicle | Small car, type A (36 %) |
| Cover | Basic — no material-damage cover (88 %) |
| Tenure | ~4 years with the company |

**In one sentence:** your highest-risk segment is young, unemployed
men driving small cars in urban areas without full cover.

### 4. Simple screening rule

For day-to-day underwriting, a simplified rule captures most of the
high-cost group without needing the full model:

> **Flag for review:** customers in areas with density > 207, aged
> under 30, without material-damage cover.

This simple rule correctly identifies 93.5 % of policies as either
standard or high-risk.

---

## What Drives Claim Costs?

The model identified the following as the strongest drivers of expected
claim cost, from most to least important:

| Factor | Effect on expected claims |
|---|---|
| **Age** | Younger customers cost significantly more — each additional year of age reduces expected cost |
| **Urban density** | Customers in densely populated areas cost more (more traffic, more accidents) |
| **Employment: Unemployed** | 2.4× higher expected cost than employed customers |
| **Employment: Retired** | Lower frequency but higher severity when they do claim |
| **Vehicle type E** | More frequent claims, but each claim tends to be cheaper |
| **No MD cover** | Associated with higher-risk profiles overall |
| **Years of tenure** | Longer-tenured customers are slightly cheaper |

These findings are consistent across both our primary model and the
independent machine-learning check.

---

## Recommended Actions

### Action matrix by customer type

Different customers need different interventions. Using the model score
alone — without checking current premium adequacy — is not sufficient.
The model tells you *who* to look at; your underwriters decide *what*
to do.

| Risk group | Recommended action |
|---|---|
| **Very High** (top 5 %) | Mandatory underwriting review at renewal. Check whether current premium covers expected claims. Consider raising deductibles, adjusting coverage limits, or repricing. Only escalate extreme cases for possible non-renewal — and only after reviewing full customer file. |
| **High** (5–20 %) | Targeted repricing or deductible adjustment. Verify that premium is adequate for the risk level. No immediate non-renewal needed. |
| **Medium** (20–50 %) | Monitor at next renewal cycle. If premium appears below the risk-adequate level, flag for selective adjustment. |
| **Low** (bottom 50 %) | Retain. These customers are your profitable base. Avoid unnecessary price increases that might push them to competitors. |

### Action by claim pattern

| Claim pattern | Why they're expensive | Best intervention |
|---|---|---|
| Frequent AND costly | Many claims, each one large | Premium increase + deductible increase + underwriting review of vehicle/use |
| Frequent but cheap | Many small claims add up | Raise deductible (eliminates small claims); consider repair-network steering |
| Rare but costly | Single large claim when it happens | Coverage/limit audit; verify vehicle valuation; check use category |

### Important guardrails

1. **Do not non-renew based solely on model score.** The model identifies
   risk, not profit. A high-risk customer paying an adequate premium is
   not unprofitable.

2. **Check current premium first.** If the customer is already paying a
   risk-adequate rate, no action is needed regardless of their score.

3. **Gender cannot be used for decisions.** EU law (since December 2012)
   prohibits gender-based pricing. Our model confirms that removing
   gender from the scoring changes results by less than 0.5 % — so
   this legal constraint costs you almost nothing in accuracy.

4. **Review extreme cases individually.** Non-renewal should only be
   considered for the most extreme outliers after a full file review,
   and only where legally and commercially justified.

---

## What We Could Not Do (and What Would Help)

| Limitation | Impact | How to fix |
|---|---|---|
| No premium data | We rank by expected claims, not by profit. A high-cost customer paying a high premium may be perfectly profitable. | Provide us with earned premium per policy — we can then calculate actual loss ratios. |
| No individual policy linkage | Frequency and severity files cannot be matched per customer. Validation is done at group level. | A single dataset with policy ID linking claims to policies would strengthen the model. |
| Dataset may be sampled | The ~50 % claim rate suggests the data is balanced, not raw. Absolute probabilities need adjustment before operational use. | Provide the full unsampled portfolio for deployment scoring. |
| No expense/commission data | We measure technical claims only, not total cost of servicing. | Including acquisition costs and admin expenses would give a fuller profitability picture. |

---

## Bottom Line

You have a small, identifiable group of customers generating
disproportionate claims. The model reliably identifies them and explains
*why* they are expensive. With premium data added in a second phase, you
could move from "high expected claims" to "confirmed loss-making" — but
even without it, the current scoring gives your underwriters a
prioritised review list that captures the right customers.

**Next steps we recommend:**

1. Score the current portfolio with the model at next renewal cycle.
2. Prioritise underwriting review for the top 5–20 % by risk score.
3. Provide premium data for a Phase 2 analysis that identifies actual
   loss-making policies.
4. Integrate the score into your renewal workflow as a flag, not an
   automatic decision.

---

*This report summarises the findings of the v5 customer profiling
analysis. Full technical documentation is available in the accompanying
notebook (`customer_profiling_v5.ipynb`).*
