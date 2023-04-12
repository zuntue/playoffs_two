import csv

# Read the results from the CSV file
results = []
with open('results.csv', 'r') as csvfile:
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

print("MAE for Spread:", mae_spread)
print("MAE for Total:", mae_total)
print("MAE for OREB:", mae_oreb)

print("\nMAE for Spread: 2nd place | MAE for Total: 1st place | MAE for OREB: 9th place | Out of a total 17 teams")
print("Note about MAE for Spread: Maybe should be 1st place too, the MAE for spread for team 11 (the 1st place team) "
      "is very suspicious.")
