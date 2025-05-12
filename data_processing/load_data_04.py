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

query = "SELECT * FROM hourly_state_with_vent_outcome_6;"  # Load a subset for initial analysis
df = pd.read_sql(query, conn)
conn.close()

# --- Step 2: Pre-process the data

# Convert hour_ts column to datetime and sort
print(df.head(10))
df['hour_ts'] = pd.to_datetime(df['hour_ts'], errors='coerce')
print(f"Number of rows before any filtering: {len(df)}")

df = df.sort_values(['stay_id', 'hour_ts']).reset_index(drop=True)

# --- Step 3: First, handle gap filtering
# For each stay_id, compute the next timestamp
df['next_time'] = df.groupby('stay_id')['hour_ts'].shift(-1)

# Calculate the gap (in hours) between the current row and the next row
df['gap_hours'] = (df['next_time'] - df['hour_ts']).dt.total_seconds() / 3600

# Create a mask for rows to drop:
# - Drop rows where the gap to the next row is > 1 hour AND the current row's device is not 'High flow nasal cannula'
# - Also drop the last row of each stay (where next_time is NaT) if its device is not 'High flow nasal cannula'
mask_drop = ((df['gap_hours'] > 1) & (df['o2_delivery_device_1'] != 'High flow nasal cannula')) | \
            ((df['next_time'].isnull()) & (df['o2_delivery_device_1'] != 'High flow nasal cannula'))

print(f"Total rows to drop from gap filtering: {mask_drop.sum()}")

# Drop the identified rows
df = df[~mask_drop].copy()

# Drop the helper columns
df.drop(columns=['next_time', 'gap_hours'], inplace=True)
print(f"Number of rows after gap filtering: {len(df)}")

# --- Step 4: Now filter for HFNC or NULL
df_clean = df[(df['o2_delivery_device_1'] == 'High flow nasal cannula') | (df['o2_delivery_device_1'].isnull())].copy()
print(f"Number of rows after filtering for HFNC or NULL: {len(df_clean)}")

# --- Step 5: Filter out stay_ids with oxygen flow > 70 L/min
flow_column = 'o2_flow'  # Adjust if your column name is different

# Get counts before filtering
unique_stays_before = df_clean['stay_id'].nunique()
rows_before = len(df_clean)

# Create a mask of stay_ids where max flow > 70
high_flow_stay_ids = df_clean.groupby('stay_id')[flow_column].max().reset_index()
high_flow_stay_ids = high_flow_stay_ids[high_flow_stay_ids[flow_column] > 70]['stay_id'].tolist()

print(f"Number of stay_ids with oxygen flow > 70 L/min: {len(high_flow_stay_ids)}")

# Filter out these stay_ids from the dataframe
df_clean = df_clean[~df_clean['stay_id'].isin(high_flow_stay_ids)]

# Get counts after filtering
unique_stays_after = df_clean['stay_id'].nunique()
rows_after = len(df_clean)

print(f"Removed {unique_stays_before - unique_stays_after} stay_ids with high flow readings")
print(f"Rows after high flow filtering: {rows_after}")

output_path = 'preprocessed_data_4.csv'
df_clean.to_csv(output_path, index=True)
print(f"Data saved to {output_path}")


# Identify flag columns (columns with 'flag' in their name)
all_columns = df_clean.columns
flag_columns = [col for col in all_columns if 'flag' in col.lower()]
non_flag_columns = [col for col in all_columns if col not in flag_columns and col != 'stay_id']

# Forward-fill non-flag columns grouped by stay_id
for col in non_flag_columns:
    df_clean[col] = df_clean.groupby('stay_id')[col].transform('ffill')

# Backward-fill non-flag columns within each stay_id group to handle missing values at the beginning
print("Performing backward fill to handle missing values at the start of each stay_id...")
for col in non_flag_columns:
    df_clean[col] = df_clean.groupby('stay_id')[col].transform('bfill')


with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    missing_values = df_clean.isnull().sum()
    print("Missing values by column after filling:")
    print(missing_values)

# Identify columns with less than 34,000 missing values
important_columns = missing_values[missing_values < 34000].index.tolist()
print(f"Keeping only columns with fewer than 34,000 missing values: {important_columns}")

# Create a new DataFrame with only the important columns
df_reduced = df_clean[important_columns]

output_path = 'preprocessed_data_4_1.csv'
df_reduced.to_csv(output_path, index=False)
print(f"Reduced data saved to {output_path}")

# Now drop rows that still have missing values
df_final = df_reduced.dropna()
print(f"Rows before final cleaning: {len(df_clean)}")
print(f"Rows after final cleaning: {len(df_final)}")
print(f"Columns retained: {len(df_final.columns)} of {len(df_clean.columns)}")

output_path = 'preprocessed_data_reduced_4.csv'
df_final.to_csv(output_path, index=False)
print(f"Reduced data saved to {output_path}")
