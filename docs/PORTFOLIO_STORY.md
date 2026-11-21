# SpendLens portfolio story

## One-sentence version
SpendLens is a personal, synthetic-data demonstration of how to build a Bayesian marketing mix model that is health-gated, calibrated with experiments, optimized under practical budget constraints, and made inspectable through Power BI and Tableau.

## The problem
Average ROI can make a large channel look healthy while hiding that the next unit of spend is inefficient. The project needed to separate historical efficiency from the next-budget decision, while keeping model uncertainty and experimental evidence visible.

## The analytical sequence
1. Fit an 8-channel Bayesian MMM across 104 weeks and 12 territories.
2. Reject an over-fit specification rather than publishing its results.
3. Tune the time trend and regional prior through a 9-run grid.
4. Calibrate two channels with territory-holdout lift tests.
5. Compare average ROI with marginal ROI.
6. Optimize a fixed INR 168.3M budget with ±20% movement limits.
7. Publish the result in Power BI and Tableau, and document the local Q&A prototype's failure modes.

## The key decision
Rep visits had average ROI 2.91 but marginal ROI 0.61. That gap—not the average alone—drove the recommendation to reduce spend. The optimizer reallocated toward paid search, paid social, and HCP email and estimated a +1.9% simulated outcome change.

## The honesty line
This is a personal project using synthetic pharma data. The +1.9% result is a model-based point estimate under constraints, not a forecast, realized lift, or client result.

## Interview prompts this project supports
- Why did you reduce the trend from 12 knots to 6?
- What changed after experiment calibration?
- Why can average ROI and marginal ROI point in different directions?
- Why constrain every channel to ±20%?
- What would you require before using this recommendation in production?
- Why publish a Q&A prototype that failed most blind-test questions?
