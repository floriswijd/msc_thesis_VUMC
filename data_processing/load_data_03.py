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

query = "SELECT * FROM hourly_state_with_vent_outcome_2;"  # Load a subset for initial analysis
df = pd.read_sql(query, conn)
conn.close()

# --- Step 2: Pre-process the data

# Convert hour_ts column to datetime and sort
print(df.head(n = 10))
df['hour_ts'] = pd.to_datetime(df['hour_ts'], errors='coerce')  # convert or coerce invalid formats to NaT
# df = df.sort_values(by='hour_ts')
# df.reset_index(drop=True, inplace=True)

print(f"Number of rows before filtering: {len(df)}")
df = df[(df['o2_delivery_device_1'] == 'High flow nasal cannula') | (df['o2_delivery_device_1'].isnull())]
print(f"Number of rows after filtering for HFNC or NULL: {len(df)}")

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

output_path = 'preprocessed_data_2.csv'
df.to_csv(output_path, index=True)
print(f"Data saved to {output_path}")

missing_values = df.isnull().sum()
print("Missing values by column after filling:")
print(missing_values)

# Identify columns with less than 34,000 missing values
important_columns = missing_values[missing_values < 34000].index.tolist()
print(f"Keeping only columns with fewer than 34,000 missing values: {important_columns}")

# Create a new DataFrame with only the important columns
df_reduced = df[important_columns]

# Now drop rows that still have missing values
df_cleaned = df_reduced.dropna()
print(f"Rows before cleaning: {len(df)}")
print(f"Rows after cleaning: {len(df_cleaned)}")
print(f"Columns retained: {len(df_cleaned.columns)} of {len(df.columns)}")

output_path = 'preprocessed_data_reduced_2.csv'
df_cleaned.to_csv(output_path, index=False)
print(f"Reduced data saved to {output_path}")
