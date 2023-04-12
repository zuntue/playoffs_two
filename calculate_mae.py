import csv

# Read the results from the CSV file
results = []
with open('pubished_results_with_mistakes.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)  # Skip the header row
    for row in csvreader:
        results.append((float(row[2]), float(row[3]), float(row[4])))

# Read the predictions from the CSV file
predictions = []
with open('predictions.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)  # Skip the header row
    for row in csvreader:
        predictions.append((float(row[2]), float(row[3]), float(row[4])))

# Calculate the Mean Absolute Error (MAE) for Spread, Total, and OREB
mae_spread = sum(abs(r[0] - p[0]) for r, p in zip(results, predictions)) / len(results)
mae_total = sum(abs(r[1] - p[1]) for r, p in zip(results, predictions)) / len(results)
mae_oreb = sum(abs(r[2] - p[2]) for r, p in zip(results, predictions)) / len(results)

print("With mistakes:")
print("MAE for Spread:", mae_spread)
print("MAE for Total:", mae_total)
print("MAE for OREB:", mae_oreb)

print("\nOut of data calculations:")
print("MAE for Spread: 13.298504727272729",
"\nMAE for Total: 17.700591818181824",
"\nMAE for OREB: 4.291665890909091")

print("\nEstimated standings: ")
print("MAE for Spread: 11th place | MAE for Total: 9th place | MAE for OREB: 9th place | Out of a total 17 teams")

# OUTSTANDING ISSUES:
# 1. the published results are wrong
# 2. the method of calculating MAE is wrong

# Note:
# I used RMSE as the loss function, and even though I scaled all three of the stats down, the scaled values for OREB
# will still have lower RMSE because they will still have been smaller, so the model will not prioritize fitting to them

# If I want my predictions to be better (potentially WAY better) I should have made a separate model for each stat, this
# would allow for the grid search for hyperparameters to tune the hyperparameters for each stat, and also tune the model
# as aggressively as possible for each stat. Other easy wins include incorporating previous head-to-head data, and some
# sort of recent performance metric, rather than just performance from the whole season. Did not realize this was
# allowed but also using 538's raptor player-based predicted spreads almost certainly could have boosted spread
# prediction accuracy tremendously. Considering I completed this 6-person, month-long, group project in effectively 24
# hours by myself, I am pretty happy with the results.
