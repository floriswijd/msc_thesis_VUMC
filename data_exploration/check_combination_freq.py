import psycopg2
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # Import seaborn for better plotting
import numpy as np # Import numpy for infinity


def check_connection():
    try:
        # Connect to the PostgreSQL database
        conn = psycopg2.connect(
            host="localhost",
            port="5432",       # default is 5432
            dbname="postgres",
            user="floppie",
            password=""
        )
        # Create a cursor object
        cur = conn.cursor()
        # Run a simple test query
        cur.execute("SELECT 1;")
        result = cur.fetchone()
        if result and result[0] == 1:
            print("Connection is working!")
        else:
            print("Unexpected result from test query.")
        cur.close()
        conn.close()
    except Exception as e:
        print("Error connecting to the database:", e)
        sys.exit(1)

# Run the connection check
# check_connection()

# Connect to the PostgreSQL database
conn = psycopg2.connect(
    host="localhost",
    port="5432",
    dbname="postgres",
    user="floppie",
    password=""
)

# Define your SQL query to retrieve the first 5 rows from the materialized view
query = "SELECT * FROM hourly_state_with_vent_outcome_2;"

# Use pandas to read the SQL query into a DataFrame
df = pd.read_sql(query, conn)

# Print the head (first few rows) of the DataFrame
print(df.head())
print("Number of rows:", len(df))

# --- Original Combination Counts and Scatter Plot ---
if 'o2_flow' in df.columns and 'fio2' in df.columns:
    # Drop rows where o2_flow or fio2 might be NaN before grouping
    df_cleaned = df.dropna(subset=['o2_flow', 'fio2'])

    combination_counts = df_cleaned.groupby(['o2_flow', 'fio2']).size().reset_index(name='count')
    print("\nCounts of Flowrate and FiO2 combinations:")
    print(combination_counts)

    # Save the combination counts to a CSV file
    output_csv_path = 'flow_fio2_combination_counts.csv'
    combination_counts.to_csv(output_csv_path, index=False)
    print(f"\nCombination counts saved to {output_csv_path}")

    # --- Plotting the distribution (Scatter Plot) ---
    try:
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            data=combination_counts,
            x='o2_flow',
            y='fio2',
            size='count',
            sizes=(20, 1000),
            alpha=0.7,
            legend='auto'
        )
        plt.title('Distribution of FiO2 and O2 Flow Combinations (Size indicates Count)')
        plt.xlabel('O2 Flow (L/min)')
        plt.ylabel('FiO2 (%)')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plot_output_path = 'flow_fio2_combination_scatter.png'
        plt.savefig(plot_output_path)
        print(f"Combination scatter plot saved to {plot_output_path}")
        # plt.show() # Optionally display plot immediately
        plt.close() # Close the plot figure
    except Exception as e:
        print(f"\nError creating scatter plot: {e}")



    # --- Binning and Plotting Binned Data ---
    print("\n--- Analyzing Binned Data ---")

    # Define bins for o2_flow
    # Bins: [0, 5), [5, 10), ..., [65, 70), [70, inf)
    flow_bins = list(range(0, 71, 5)) + [np.inf]
    flow_labels = [f'{i}-{i+5}' for i in range(0, 66, 5)] + ['>70']

    # Define bins for fio2
    # Assuming FiO2 is 0-100. Bins: [0, 5), [5, 10), ..., [95, 100]
    # Adjust max value if FiO2 can exceed 100, or min if below 0
    fio2_bins = list(range(0, 101, 5))
    fio2_labels = [f'{i}-{i+5}' for i in range(0, 96, 5)]

    # Create binned columns - use the cleaned dataframe
    # Use right=False to make bins like [0, 5), [5, 10), etc.
    # Add include_lowest=True for the first bin if needed, depends on data range
    try:
        df_cleaned['o2_flow_bin'] = pd.cut(df_cleaned['o2_flow'], bins=flow_bins, labels=flow_labels, right=False)
        # For FiO2, the last bin should include 100, so right=True (default) is okay if max is 100
        # If using right=False, the last label needs adjustment or handle 100 separately
        df_cleaned['fio2_bin'] = pd.cut(df_cleaned['fio2'], bins=fio2_bins, labels=fio2_labels, right=False, include_lowest=True) # include_lowest=True makes the first bin [0, 5]

        # Handle potential NaNs created by binning if values fall outside bins
        df_binned = df_cleaned.dropna(subset=['o2_flow_bin', 'fio2_bin'])

        # Calculate counts for binned combinations
        binned_counts = df_binned.groupby(['o2_flow_bin', 'fio2_bin'], observed=False).size().reset_index(name='count')
        print("\nCounts of Binned Flowrate and FiO2 combinations:")
        # Filter out zero counts before printing if desired
        print(binned_counts[binned_counts['count'] > 0])

        # Save binned counts
        binned_output_csv_path = 'binned_flow_fio2_counts.csv'
        binned_counts.to_csv(binned_output_csv_path, index=False)
        print(f"\nBinned combination counts saved to {binned_output_csv_path}")

        # Filter out combinations with zero counts for plotting
        binned_counts_filtered = binned_counts[binned_counts['count'] > 0]

        # --- Plotting Binned Data (Heatmap) ---
        if not binned_counts_filtered.empty:
            try:
                # Pivot for heatmap
                # Ensure all categories are present for consistent heatmap structure
                binned_pivot = binned_counts_filtered.pivot_table(index='fio2_bin', columns='o2_flow_bin', values='count', fill_value=0)
                # Reindex to ensure all defined labels are present, even if they had 0 count originally
                binned_pivot = binned_pivot.reindex(index=fio2_labels, columns=flow_labels, fill_value=0)


                plt.figure(figsize=(14, 10))
                sns.heatmap(binned_pivot, annot=True, fmt=".0f", cmap="viridis", linewidths=.5, cbar_kws={'label': 'Count'})
                plt.title('Distribution of Binned FiO2 and O2 Flow Combinations')
                plt.xlabel('O2 Flow Bin (L/min)')
                plt.ylabel('FiO2 Bin (%)')
                plt.xticks(rotation=45, ha='right') # Rotate x-axis labels for better readability
                plt.yticks(rotation=0)
                plt.tight_layout()
                binned_plot_output_path = 'binned_flow_fio2_heatmap.png'
                plt.savefig(binned_plot_output_path)
                print(f"Binned combination heatmap saved to {binned_plot_output_path}")
                # plt.show() # Optionally display plot immediately
                plt.close() # Close the plot figure
            except Exception as e:
                print(f"\nError creating binned heatmap: {e}")

            # --- Plotting Distribution of Counts (Histogram) ---
            try:
                plt.figure(figsize=(10, 6))
                # Use histplot to show the distribution of the 'count' values
                sns.histplot(data=binned_counts_filtered, x='count', bins=30, kde=False) # Adjust bins as needed
                plt.title('Distribution of Counts per Binned Combination')
                plt.xlabel('Number of Occurrences (Count)')
                plt.ylabel('Number of Bin Combinations')
                plt.yscale('log') # Use log scale for y-axis if counts vary widely
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                count_dist_plot_path = 'binned_count_distribution_hist.png'
                plt.savefig(count_dist_plot_path)
                print(f"Count distribution histogram saved to {count_dist_plot_path}")
                # plt.show() # Optionally display plot immediately
                plt.close() # Close the plot figure
            except Exception as e:
                print(f"\nError creating count distribution histogram: {e}")

        else:
            print("\nNo data available for binned heatmap or count distribution plot after processing.")

    except ValueError as ve:
        print(f"\nError during binning: {ve}")
        print("Check if data in 'o2_flow' or 'fio2' falls outside the defined bin ranges.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during binning or plotting: {e}")


else:
    print("\nError: 'o2_flow' or 'fio2' column not found in the DataFrame.")
    print("Available columns:", df.columns.tolist())

# Close the database connection
conn.close()