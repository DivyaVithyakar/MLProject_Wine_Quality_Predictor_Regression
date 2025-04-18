import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from functions import  quanQual,univaiate,replace_outlier
import seaborn as sns
import matplotlib.pyplot as plt

# Load and clean data
df = pd.read_csv("../data/winequality-red.csv", sep=';')

#Check empty values
print(df.isnull().sum())

#quan, Qual
quan, qual = quanQual(df)
print(quan,qual)

#Univariate
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)

uni_describe = univaiate(df, quan)
print(uni_describe)

#Replace Outliers
df = replace_outlier(df, quan, uni_describe)
# Based on earlier skewness, or automate based on skew > 1
log_transform_cols = ['residual sugar', 'chlorides', 'sulphates']
for col in log_transform_cols:
    df[col] = np.log1p(df[col])  # log(1 + x) handles zeros safely


#View transformed summary
transformed_stats = univaiate(df, quan)
print("Transformed Summary:\n", transformed_stats)

# Save the cleaned dataset to a new CSV file
df.to_csv("../data/winequality-red-cleaned.csv", index=False)

# Compute correlation matrix
correlation_matrix = df.corr()

# Set the figure size
plt.figure(figsize=(12, 8))

# Create heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

# Title and display
plt.title("📊 Feature Correlation Heatmap - Wine Quality Dataset")
plt.tight_layout()
plt.show()