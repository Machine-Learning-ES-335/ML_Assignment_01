# HAR       # Task 1 EDA

Answer 1. The six activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING) show clear patterns in the waveform plots.
- Dynamic activities (walking variants) exhibit periodic oscillations with varying frequency and amplitude:
- WALKING → regular sinusoidal patterns.
- WALKING_UPSTAIRS → similar but with slightly higher amplitude and irregular steps.
- WALKING_DOWNSTAIRS → higher peaks due to stronger impacts.
- Static activities (SITTING, STANDING, LAYING) are mostly flat with very low variance.

These patterns suggest that a machine learning model can easily learn these differences because dynamic vs. static activities have distinct signatures.
<img width="594" height="252" alt="ML Intertial signals shape" src="https://github.com/user-attachments/assets/b339232a-34df-4501-a82d-6692e591ed7a" />

Answer 2.
The magnitude of linear acceleration: $\sqrt{acc_x^2 + acc_y^2 + acc_z^2}$ shows clear separation between static and dynamic activities.
- Dynamic activities have high magnitude and variability; static ones are nearly constant.
Thus, a simple threshold rule could separate static vs. dynamic.
- However, for finer classification (e.g., sitting vs standing or walking vs walking_upstairs), machine learning is required to capture subtle patterns.

Q3. (a) Use PCA on total acceleration and plot a scatter plot.
Answer 3.a:
- PCA projection onto 2D shows clusters for dynamic activities separated from static ones.
- LAYING, SITTING, and STANDING form a tight cluster (similar acceleration magnitude), while WALKING, WALKING_UPSTAIRS, and WALKING_DOWNSTAIRS spread apart due to higher variability.
- However, overlap exists among the three dynamic activities, indicating raw acceleration alone isn’t enough for perfect class separation.

Q3 (b) Use TSFEL features + PCA for visualization.
Answer 3.b:
- Extracting time-domain and frequency-domain features (mean, standard deviation, energy, spectral entropy, etc.) with TSFEL improves clustering.
PCA scatter plot now shows:
- Clearer separation between WALKING and WALKING_DOWNSTAIRS due to spectral differences.
- Static activities remain tightly grouped but are slightly more separable (LAYING vs. STANDING).
This shows engineered features carry richer discriminatory information than raw acceleration.

Q3 (c)  PCA with original dataset features
Using features from X_train.txt (561 features) with PCA.

Answer:
- PCA scatter plot of provided features produces the clearest separation:
>- Dynamic classes (WALKING, UPSTAIRS, DOWNSTAIRS) form well-defined clusters.
>- Static classes (LAYING, SITTING, STANDING) are also separable with minor overlaps.
- This is expected since these features are already well-engineered statistical and frequency-domain descriptors.

Answer 3.d: Comparison of PCA methods

Feature Set	           vs              Observations

Total Acceleration ==>	Rough static-vs-dynamic split; poor fine-grained separation.

TSFEL Features	==> Better spread, captures temporal-frequency info; clusters are more distinct.

Provided Features	==> Best separation; features already optimized for HAR tasks.
<img width="827" height="1000" alt="Screenshot 2025-08-27 200333" src="https://github.com/user-attachments/assets/fe015dc3-48da-4675-9f27-534096438ec0" />
Q4. Calculate correlation of TSFEL features and dataset features. Identify redundancy.

Answer: Correlation matrices reveal:

>- Several highly correlated features (e.g., mean and median, or signal energy and RMS in TSFEL).
>- In dataset features, some frequency components are also redundant.

- Redundant features do not add new information and can be removed or reduced via PCA or feature selection for faster and more stable models.

**Conclusion:** PCA on provided dataset features is best for visualization and likely better for downstream classification.




<img width="913" height="912" alt="ML TSFEL features shape" src="https://github.com/user-attachments/assets/659e43fb-6daf-4a77-9da8-e30bdf4241da" />

# Task 2 Desicion Trees for HAR
1. a. Observations: High accuracy for dynamic activities (walking, walking upstairs/downstairs)
Static postures (sitting vs standing) confused sometimes because raw data alone doesn’t clearly differentiate them.
<img width="760" height="256" alt="ML Raw vs Provided vs TSFEL" src="https://github.com/user-attachments/assets/b7001855-91a4-4065-99a0-ddf7ebd38513" />

b.

c.

d.

2.

3.

