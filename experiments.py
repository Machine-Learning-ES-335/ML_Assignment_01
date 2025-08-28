import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
import os

np.random.seed(42)
num_avg = 3  # small repetitions
N_vals = [100, 500, 1000]  # small datasets for fast run
M_vals = [5, 10, 20]      
results_file = "runtime_results.csv"
plots_dir = "plots"
md_file = "runtime_results.md"
os.makedirs(plots_dir, exist_ok=True)

# ------------------ Generate data for each case ------------------
def generate_data_case(N, M, case="DiDo"):
    if case=="DiDo": X,y = np.random.randint(0,2,(N,M)), np.random.randint(0,2,N)
    elif case=="DiRo": X,y = np.random.randint(0,2,(N,M)), np.random.randn(N)
    elif case=="RiDo": X,y = np.random.randn(N,M), np.random.randint(0,2,N)
    elif case=="RiRo": X,y = np.random.randn(N,M), np.random.randn(N)
    else: raise ValueError(f"Unknown case: {case}")
    return X,y

# ------------------ Measure runtime ------------------
def measure_runtime(tree_type, N_vals, M_vals, case):
    results=[]
    for N in N_vals:
        for M in M_vals:
            fit_times, pred_times = [],[]
            for _ in range(num_avg):
                X,y = generate_data_case(N,M,case)
                dt = DecisionTree(criterion=tree_type)
                # Fit
                t0=time.time(); dt.fit(X,pd.Series(y)); fit_times.append(time.time()-t0)
                # Predict
                X_test,_=generate_data_case(N//10,M,case)
                t0=time.time(); dt.predict(X_test); pred_times.append(time.time()-t0)
            results.append({
                "tree_type":tree_type,"case":case,"N":N,"M":M,
                "fit_mean":np.mean(fit_times),"fit_std":np.std(fit_times),
                "predict_mean":np.mean(pred_times),"predict_std":np.std(pred_times)
            })
            print(f"{tree_type} | {case} | N={N}, M={M} | Fit:{np.mean(fit_times):.4f}s | Predict:{np.mean(pred_times):.4f}s")
    return results

# ------------------ Plot results ------------------
def plot_results(results, case):
    df=pd.DataFrame(results)
    plt.figure(figsize=(12,5))
    # Fit
    plt.subplot(1,2,1)
    for N in df['N'].unique():
        dfN=df[df['N']==N]
        plt.plot(dfN['M'], dfN['fit_mean'], marker='o', label=f"N={N}")
    plt.xlabel("M"); plt.ylabel("Fit Time (s)"); plt.title(f"Fit Time ({case})"); plt.legend()
    # Predict
    plt.subplot(1,2,2)
    for N in df['N'].unique():
        dfN=df[df['N']==N]
        plt.plot(dfN['M'], dfN['predict_mean'], marker='o', label=f"N={N}")
    plt.xlabel("M"); plt.ylabel("Predict Time (s)"); plt.title(f"Predict Time ({case})"); plt.legend()
    plt.tight_layout()
    plot_path=os.path.join(plots_dir,f"{case}_runtime.png")
    plt.savefig(plot_path); plt.close()
    return plot_path

# ------------------ Run experiments ------------------
all_results=[]
cases=["DiDo","DiRo","RiDo","RiRo"]
for case in cases:
    tree_types = ["information_gain","gini_index"]
    if case in ["DiRo","RiRo"]: tree_types=["information_gain"]  # gini only for discrete output
    for t in tree_types:
        res = measure_runtime(t,N_vals,M_vals,case)
        all_results.extend(res)
        plot_path = plot_results(res,case)

# ------------------ Save results ------------------
df_results=pd.DataFrame(all_results)
df_results.to_csv(results_file,index=False)
print(f"Results saved to {results_file}")

# ------------------ Create markdown ------------------
with open(md_file,"w") as f:
    f.write("# Decision Tree Runtime Experiments\n\n")
    f.write("## Plots\n")
    for case in cases:
        f.write(f"### {case}\n")
        f.write(f"![{case}]({plots_dir}/{case}_runtime.png)\n\n")
print(f"Markdown saved to {md_file}")
