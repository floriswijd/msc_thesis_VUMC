import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Step 1: Connect to PostgreSQL and load the data
conn = psycopg2.connect(
    host="localhost",
    port="5432",
    dbname="postgres",
    user="floppie",
    password=""
)

query = "SELECT * FROM hourly_state_with_vent_outcome;"  # Load a subset for initial analysis
df = pd.read_sql(query, conn)
conn.close()

# --- Step 2: Pre-process the data

# Convert hour_ts column to datetime and sort
print(df.head(n = 10))
df['hour_ts'] = pd.to_datetime(df['hour_ts'], errors='coerce')  # convert or coerce invalid formats to NaT
# df = df.sort_values(by='hour_ts')
# df.reset_index(drop=True, inplace=True)

# Checking for missing values
missing_values = df.isnull().sum()
print("Missing values by column:")
print(missing_values)

# Ensure the data is sorted by stay_id and hour_ts:
df = df.sort_values(['stay_id', 'hour_ts'])

# Identify flag columns (columns with 'flag' in their name)
all_columns = df.columns
flag_columns = [col for col in all_columns if 'flag' in col.lower()]
non_flag_columns = [col for col in all_columns if col not in flag_columns and col != 'stay_id']

for col in non_flag_columns:
    df[col] = df.groupby('stay_id')[col].transform('ffill')

# Backward fill non-flag columns within each stay_id group to handle missing values at the beginning
print("Performing backward fill to handle missing values at the start of each stay_id...")
for col in non_flag_columns:
    df[col] = df.groupby('stay_id')[col].transform('bfill')

output_path = 'preprocessed_data.csv'
df.to_csv(output_path, index=True)
print(f"Data saved to {output_path}")

# --- Step 3: Exploratory Data Analysis (EDA) plots

# Plot settings
plt.style.use('default')  # choose your style
num_columns = df.select_dtypes(include=['float64', 'int64']).columns

missing_values = df.isnull().sum()
print("Missing values by column:")
print(missing_values)
# Print the first few rows of the DataFrame
print(df.head())
print("Number of rows:", len(df))
# Print the data types of each column
# print("Data types of each column:")
# print(df.dtypes)

# Create histograms and boxplots for each numeric column
# for col in num_columns:
#     fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
#     # Histogram
#     axes[0].hist(df[col].dropna(), bins=30, edgecolor='black')
#     axes[0].set_title(f'Histogram of {col}')
#     axes[0].set_xlabel(col)
#     axes[0].set_ylabel('Frequency')
    
#     # Boxplot
#     axes[1].boxplot(df[col].dropna(), vert=False)
#     axes[1].set_title(f'Boxplot of {col}')
#     axes[1].set_xlabel(col)
    
#     plt.tight_layout()
#     plt.show()

# # Time-Series Plot for a selected vital sign (example: heart_rate)
# plt.figure(figsize=(12, 6))
# plt.plot(df['hour_ts'], df['heart_rate'], marker='o', linestyle='-')
# plt.title('Heart Rate over Time')
# plt.xlabel('Time')
# plt.ylabel('Heart Rate')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# # Optional: Visualize missing data pattern using a heatmap
# plt.figure(figsize=(14, 8))
# sns.heatmap(df.isnull(), cbar=False)
# plt.title('Missing Values Heatmap')
# plt.show()
