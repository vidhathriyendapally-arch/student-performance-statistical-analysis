import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, norm

# ---------------------------------------
# Load the two datasets
# ---------------------------------------
df1 = pd.read_csv("StudentPerformanceFactors.csv")
df2 = pd.read_csv("Exam_Score_Prediction.csv")

# ---------------------------------------
# Select the common numeric column
# ---------------------------------------
data1 = df1["exam_score"].dropna()
data2 = df2["Exam_Score"].dropna()

print("Sample sizes:", len(data1), len(data2))

# ---------------------------------------
# 1️⃣ F-TEST (Variance Comparison)
# ---------------------------------------
var1 = np.var(data1, ddof=1)
var2 = np.var(data2, ddof=1)

F = var1 / var2
dfn = len(data1) - 1
dfd = len(data2) - 1

print("\n--------- F TEST ---------")
print("Variance 1:", var1)
print("Variance 2:", var2)
print("F-value:", F)
print("Degrees of freedom:", dfn, dfd)

# ---------------------------------------
# 2️⃣ T-TEST (Mean Comparison)
# ---------------------------------------
t_stat, p_value = ttest_ind(data1, data2)

print("\n--------- T TEST ---------")
print("t-statistic:", t_stat)
print("p-value:", p_value)

if p_value < 0.05:
    print("Conclusion: Means are significantly different.")
else:
    print("Conclusion: Means are NOT significantly different.")

# ---------------------------------------
# 3️⃣ Z-TEST (Large Sample Mean Comparison)
# ---------------------------------------
mean1 = np.mean(data1)
mean2 = np.mean(data2)

std1 = np.std(data1, ddof=1)
std2 = np.std(data2, ddof=1)

n1 = len(data1)
n2 = len(data2)

z_num = (mean1 - mean2)
z_den = np.sqrt((std1**2 / n1) + (std2**2 / n2))
z_value = z_num / z_den

p_z = 2 * (1 - norm.cdf(abs(z_value)))

print("\n--------- Z TEST ---------")
print("Z-value:", z_value)
print("p-value:", p_z)

if p_z < 0.05:
    print("Conclusion: Means are significantly different.")
else:
    print("Conclusion: Means are NOT significantly different.")
