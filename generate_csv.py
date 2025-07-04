import pandas as pd
import numpy as np

# List of Indian states (10 examples)
states = ['MH', 'DL', 'KA', 'TN', 'GJ', 'RJ', 'UP', 'WB', 'PB', 'HR']

# Generate monthly dates from Jan to Jun 2025
months = pd.date_range(start="2025-01-01", end="2025-06-01", freq="MS")

# Create data
data = []
np.random.seed(42)
for month in months:
    for state in states:
        monthly_cases = np.random.randint(40, 130)  # Realistic small case numbers
        data.append({
            "Date": month.strftime("%Y-%m"),
            "State": state,
            "Monthly_Cases": monthly_cases
        })

# Convert to DataFrame and save
df = pd.DataFrame(data)
df.to_csv("state_wise_monthly_2025.csv", index=False)
print("✅ CSV file 'state_wise_monthly_2025.csv' created successfully!")
