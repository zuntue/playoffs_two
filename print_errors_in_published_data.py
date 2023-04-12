import csv

def read_csv(file_name):
    data = []
    with open(file_name, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            data.append(row)
    return data

# Read the data from the two CSV files
results_with_mistakes = read_csv('pubished_results_with_mistakes.csv')[1:]  # [1:] to exclude the header
correct_results = read_csv('results.csv')[1:]

# Check if the CSV files have the same number of rows
if len(results_with_mistakes) != len(correct_results):
    print("The CSV files have a different number of rows.")
else:
    # Compare the rows and print discrepancies
    for i, (row_with_mistakes, correct_row) in enumerate(zip(results_with_mistakes, correct_results)):
        discrepancies = []
        for j, (cell_with_mistakes, correct_cell) in enumerate(zip(row_with_mistakes, correct_row)):
            try:
                if float(cell_with_mistakes) != float(correct_cell):
                    discrepancies.append((j, cell_with_mistakes, correct_cell))
            except ValueError:
                if cell_with_mistakes != correct_cell:
                    discrepancies.append((j, cell_with_mistakes, correct_cell))

        if discrepancies:
            print(f"Row {i + 2} has discrepancies:")
            for col, cell_with_mistakes, correct_cell in discrepancies:
                print(f"  Column {col + 1}: '{cell_with_mistakes}' should be '{correct_cell}'")