-------------------------------------------automative efficiency--------------------------------------------
Custom Decision Tree Performance:
RMSE: 3.5179
MAE: 2.7084

scikit-learn Decision Tree Performance:
RMSE: 3.5058
MAE: 2.4288

Custom decision tree is working correctly and gives comparable results to a well tested library like sklearn.

# Decision Tree Runtime Experiments

### DiDo

![DiDo](plots/DiDo_runtime.png)

### DiRo

![DiRo](plots/DiRo_runtime.png)

### RiDo

![RiDo](plots/RiDo_runtime.png)

### RiRo

![RiRo](plots/RiRo_runtime.png)

Time Complexity Analysis

When building a decision tree, the algorithm looks for the best split at each node and recursively divides the data until leaf nodes are reached. For N samples and M features, this means the training time grows roughly like
𝑂(𝑁⋅𝑀⋅log𝑁)

Prediction is faster because each sample only travels from the root to a leaf, so the complexity is about
𝑂(𝑃⋅log𝑁) where P is the number of test samples.

From our experiments:

For all four cases—DiDo (discrete→discrete), DiRo (discrete→real), RiDo (real→discrete), RiRo (real→real)—we see that fit time increases with both N and M, just like theory predicts.

Real-valued inputs or outputs (DiRo, RiDo, RiRo) make fitting a little slower due to numeric comparisons, but overall the trend matches what we expect from the theory.
