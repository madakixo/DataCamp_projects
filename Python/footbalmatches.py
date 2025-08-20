# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import pingouin as pg
from scipy.stats import mannwhitneyu

# Load men's and women's football match datasets
men = pd.read_csv("men_results.csv")
women = pd.read_csv("women_results.csv")

# Convert 'date' column to datetime for filtering
men["date"] = pd.to_datetime(men["date"], errors='coerce')  # Handle invalid dates
women["date"] = pd.to_datetime(women["date"], errors='coerce')

# Filter data for FIFA World Cup matches after January 1, 2002
men_subset = men[(men["date"] > "2002-01-01") & (men["tournament"] == "FIFA World Cup")]
women_subset = women[(women["date"] > "2002-01-01") & (women["tournament"] == "FIFA World Cup")]

# Add 'group' column to identify men's and women's matches
men_subset = men_subset.assign(group="men")
women_subset = women_subset.assign(group="women")

# Calculate total goals scored per match (home + away scores)
men_subset = men_subset.assign(goals_scored=men_subset["home_score"] + men_subset["away_score"])
women_subset = women_subset.assign(goals_scored=women_subset["home_score"] + women_subset["away_score"])

# Visualize distribution of goals scored to assess normality
plt.figure(figsize=(8, 6))  # Set figure size for better readability
men_subset["goals_scored"].hist(bins=20, alpha=0.5, label="Men")
women_subset["goals_scored"].hist(bins=20, alpha=0.5, label="Women")
plt.xlabel("Goals Scored")
plt.ylabel("Frequency")
plt.title("Distribution of Goals Scored in FIFA World Cup Matches")
plt.legend()
plt.show()
plt.clf()  # Clear the figure to free memory

# Combine men's and women's data for analysis
both = pd.concat([women_subset, men_subset], axis=0, ignore_index=True)

# Select relevant columns for statistical test
both_subset = both[["goals_scored", "group"]]

# Perform Mann-Whitney U test using pingouin (right-tailed)
results_pg = pg.mwu(x=both_subset[both_subset["group"] == "women"]["goals_scored"],
                    y=both_subset[both_subset["group"] == "men"]["goals_scored"],
                    alternative="greater")

# Alternative: Perform Mann-Whitney U test using SciPy (right-tailed)
results_scipy = mannwhitneyu(x=women_subset["goals_scored"],
                             y=men_subset["goals_scored"],
                             alternative="greater",
                             method="auto")  # Auto method for efficiency

# Extract p-value from pingouin results
p_val = results_pg["p-val"].iloc[0]

# Determine hypothesis test outcome using significance level (alpha = 0.01)
result = "reject" if p_val <= 0.01 else "fail to reject"

# Store results in a dictionary
result_dict = {"p_val": p_val, "result": result}

# Display the results
print("\nMann-Whitney U Test Results:")
print(f"p-value: {p_val:.4f}")
print(f"Result: {result}")
print("\nResult Dictionary:", result_dict)
