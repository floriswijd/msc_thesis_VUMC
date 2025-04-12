import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the preprocessed data
df = pd.read_csv('preprocessed_data_reduced_3.csv')

# Display basic descriptive statistics for the columns of interest
print("Descriptive statistics for SpO₂, Respiratory Rate, FiO₂, and O₂ flow:")
print(df[['spo2', 'resp_rate', 'fio2', 'o2_flow']].describe())

# ----------------------------
# Exploratory Plots for Each Variable
# ----------------------------
plt.figure(figsize=(18, 4))

# SpO₂ histogram
plt.subplot(1, 4, 1)
plt.hist(df['spo2'], bins=30, color='skyblue', edgecolor='black')
plt.xlabel('SpO₂')
plt.ylabel('Frequency')
plt.title('Distribution of SpO₂')

# Respiratory Rate histogram
plt.subplot(1, 4, 2)
plt.hist(df['resp_rate'], bins=30, color='salmon', edgecolor='black')
plt.xlabel('Respiratory Rate')
plt.ylabel('Frequency')
plt.title('Distribution of Respiratory Rate')

# FiO₂ histogram
plt.subplot(1, 4, 3)
plt.hist(df['fio2'], bins=30, color='lightgreen', edgecolor='black')
plt.xlabel('FiO₂')
plt.ylabel('Frequency')
plt.title('Distribution of FiO₂')

# O₂ flow histogram
plt.subplot(1, 4, 4)
plt.hist(df['o2_flow'], bins=30, color='orchid', edgecolor='black')
plt.xlabel('O₂ Flow')
plt.ylabel('Frequency')
plt.title('Distribution of O₂ Flow')

plt.tight_layout()
plt.show()

# ----------------------------
# Boxplots for Outlier Detection
# ----------------------------
plt.figure(figsize=(18, 4))

plt.subplot(1, 4, 1)
sns.boxplot(x=df['spo2'], color='skyblue')
plt.title('Boxplot of SpO₂')

plt.subplot(1, 4, 2)
sns.boxplot(x=df['resp_rate'], color='salmon')
plt.title('Boxplot of Respiratory Rate')

plt.subplot(1, 4, 3)
sns.boxplot(x=df['fio2'], color='lightgreen')
plt.title('Boxplot of FiO₂')

plt.subplot(1, 4, 4)
sns.boxplot(x=df['o2_flow'], color='orchid')
plt.title('Boxplot of O₂ Flow')

plt.tight_layout()
plt.show()

# ----------------------------
# Define the Discretized State Space
# ----------------------------

# 1. SpO₂ binning based on clinical ranges:
#    - Below 88% (hypoxia)
#    - 88–91%
#    - 92–96% (optimal range)
#    - Above 96% (potential hyperoxia)
spo2_bins = [-np.inf, 88, 92, 97, np.inf]
spo2_labels = ["<88", "88-91", "92-96", ">96"]
df['spo2_bin'] = pd.cut(df['spo2'], bins=spo2_bins, labels=spo2_labels)
print("\nSpO₂ bin counts:")
print(df['spo2_bin'].value_counts())

# 2. FiO₂ binning using increments of 5%
fio2_min = df['fio2'].min()
fio2_max = df['fio2'].max()
# Adjusting start to nearest multiple of 5
fio2_bins = np.arange(fio2_min - (fio2_min % 5), fio2_max + 5, 5)
df['fio2_bin'] = pd.cut(df['fio2'], bins=fio2_bins)
print("\nFiO₂ bin counts:")
print(df['fio2_bin'].value_counts())

# 3. Respiratory Rate binning: Use quantiles as an initial strategy
resp_bins = [-np.inf] + list(df['resp_rate'].quantile([0.25, 0.5, 0.75])) + [np.inf]
df['resp_rate_bin'] = pd.cut(df['resp_rate'], bins=resp_bins)
print("\nRespiratory Rate bin counts:")
print(df['resp_rate_bin'].value_counts())

# 4. O₂ Flow binning: Create bins using increments of 5 (or adjust based on clinical relevance)
o2_flow_min = df['o2_flow'].min()
o2_flow_max = df['o2_flow'].max()
# Determine a suitable increment value; here, using increments of 5
flow_increment = 5
o2_flow_bins = np.arange(o2_flow_min - (o2_flow_min % flow_increment), o2_flow_max + flow_increment, flow_increment)
df['o2_flow_bin'] = pd.cut(df['o2_flow'], bins=o2_flow_bins)
print("\nO₂ Flow bin counts:")
print(df['o2_flow_bin'].value_counts())

# ----------------------------
# Visualize the Binned Data: Count Plots
# ----------------------------
plt.figure(figsize=(20, 4))

plt.subplot(1, 4, 1)
sns.countplot(data=df, x='spo2_bin', palette='Blues')
plt.title('SpO₂ Bins')

plt.subplot(1, 4, 2)
sns.countplot(data=df, x='resp_rate_bin', palette='Reds')
plt.title('Respiratory Rate Bins')

plt.subplot(1, 4, 3)
sns.countplot(data=df, x='fio2_bin', palette='Greens')
plt.title('FiO₂ Bins')

plt.subplot(1, 4, 4)
sns.countplot(data=df, x='o2_flow_bin', palette='Purples')
plt.title('O₂ Flow Bins')

plt.tight_layout()
plt.show()