# Representation-controlled optimizer comparison

The Milestone 1 methods search different spaces. GP searches symbolic trees, while GA, PSO, and ES search masks plus linear and squared synthetic combinations. Comparing their test scores does not isolate the quality of their search algorithms.

For the next optimizer study, use the same candidate representation, decoder, validation MAE, downstream regression model, training-only preprocessing, initialization distribution, 1,000-evaluation budget, and outer/inner splits for every optimizer. The existing GA, PSO, and ES operators can be adapted to this shared representation; new algorithms such as DE and CMA-ES should use it too. Report GP separately as a symbolic representation family. Keep the current method-specific results as a reference, with a distinct protocol name in every output directory.

The held-out diagnostics show that squared constructed features can extrapolate sharply on California Housing. A separate robustness study should compare the current linear downstream fit against training-only normalization plus Ridge and, if needed, training-derived feature clipping. It must tune any regularization or clipping on inner data only and report held-out MAE, RMSE, R², and extreme residuals. Changing the downstream fit inside the current equal-budget study would mix two methodological changes in one comparison.
