# Original protocol: extreme prediction diagnostics

`replay_diagnostics.py` reconstructed every saved optimizer feature definition and refitted its final linear model on the recorded training split. Every replayed held-out MAE matched `raw_results.csv` within 1e-6. The script saved the five largest absolute residuals per run in `extreme_residuals.csv` and generated-feature ranges and coefficients in `feature_diagnostics.csv`.

The largest California Housing error was PSO construction on outer split 1, optimizer seed 101, test row 19006: actual target 1.375, prediction 558.631, absolute error 557.256. One synthetic feature on that row was 17,791.597, with downstream coefficient 0.30438. Its absolute value on the test set was about 7,948 times the training 99th percentile for that feature. Other features partly offset its contribution, but the final prediction was still extreme.

Across California runs, the maximum absolute residual was 557.256 for PSO construction, 326.377 for PSO combined, and 308.598 for GA construction. This explains why some construction variants had modest MAE but poor RMSE and R². The diagnostics identify extrapolation from unusually large constructed feature values, especially squared combinations. They do not establish that regularization alone would fix it; that requires an inner-validated robustness experiment with the same held-out test discipline.
