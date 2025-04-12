import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("hel;lloo")

# --- Step 1: Connect to PostgreSQL and load the data
conn = psycopg2.connect(
    host="localhost",
    port="5432",
    dbname="postgres",
    user="floppie",
    password=""
)

query = "SELECT * FROM hourly_state_with_vent_outcome_2;"
df = pd.read_sql(query, conn)
conn.close()

print(f"Total number of rows in dataset: {len(df)}")

# Convert hour_ts column to datetime
df['hour_ts'] = pd.to_datetime(df['hour_ts'], errors='coerce')

# --- Step 2: Analyze duplicate combinations of stay_id and hour_ts
duplicate_count = df.duplicated(subset=['stay_id', 'hour_ts']).sum()
print(f"Number of duplicate (stay_id, hour_ts) combinations: {duplicate_count}")

# Find all rows that are part of duplicated combinations
duplicates = df[df.duplicated(subset=['stay_id', 'hour_ts'], keep=False)]
print(f"Number of rows involved in duplications: {len(duplicates)}")

# Count how many duplicates each stay_id has
if len(duplicates) > 0:
    duplicate_counts_by_stay = duplicates.groupby('stay_id').size()
    print(f"Number of unique patients with duplicates: {len(duplicate_counts_by_stay)}")
    print("\nDuplicate distribution:")
    print(duplicate_counts_by_stay.value_counts())

    # Show examples of duplicated records
    print("\nExample of duplicated records:")
    example_stay = duplicates['stay_id'].iloc[0]
    example_hour = duplicates['hour_ts'].iloc[0]
    example = df[(df['stay_id'] == example_stay) & (df['hour_ts'] == example_hour)]
    print(example[['stay_id', 'hour_ts', 'heart_rate', 'spo2', 'o2_delivery_device_1']].head(5))
    
    # Check if values differ in the duplicates
    print("\nAnalyzing differences in duplicate rows...")
    
    # Group by stay_id and hour_ts and check if any columns have different values
    for col in df.columns:
        if col not in ['stay_id', 'hour_ts']:
            # Count how many unique values each stay_id, hour_ts combination has for this column
            value_counts = duplicates.groupby(['stay_id', 'hour_ts'])[col].nunique()
            # Filter to only those with more than one unique value
            differing_values = value_counts[value_counts > 1]
            if len(differing_values) > 0:
                print(f"Column '{col}' has differing values in {len(differing_values)} duplicate combinations")
                
                # Show an example
                example_stay_hour = differing_values.index[0]
                example_diff = df[(df['stay_id'] == example_stay_hour[0]) & 
                                 (df['hour_ts'] == example_stay_hour[1])][[col]]
                print(f"Example for stay_id {example_stay_hour[0]}, hour_ts {example_stay_hour[1]}:")
                print(example_diff)
    
    # --- Step 3: Options for handling duplicates
    print("\nOptions for handling duplicates:")
    print("1. Keep first occurrence")
    print("2. Keep last occurrence")
    print("3. Aggregate numeric values (mean)")
    print("4. Drop all duplicate occurrences")
    
    # Example of implementing option 1 (keep first occurrence)
    df_deduplicated = df.drop_duplicates(subset=['stay_id', 'hour_ts'], keep='first')
    print(f"\nAfter deduplication (keeping first): {len(df_deduplicated)} rows")
else:
    print("No duplicates found in the dataset.")

# Save the deduplicated data to CSV for inspection
output_path = 'deduplicated_data.csv'
df_deduplicated.to_csv(output_path, index=False)
print(f"Deduplicated data saved to {output_path}")