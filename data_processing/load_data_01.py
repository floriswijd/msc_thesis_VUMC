import psycopg2
import sys
import pandas as pd

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
query = "SELECT * FROM hourly_state_with_vent_outcome_6;"

# Use pandas to read the SQL query into a DataFrame
df = pd.read_sql(query, conn)

# Print the head (first few rows) of the DataFrame
print(df.head())
print("Number of rows:", len(df))

output_path = 'input_data.csv'
df.to_csv(output_path, index=True)
print(f"Data saved to {output_path}")
# Close the database connection
conn.close()
