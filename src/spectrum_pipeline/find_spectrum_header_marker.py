import os

SPECTRUM_FOLDER = r"C:\STUDIEZZZ\Semester Thesis\Semester Project\CSV File data 10-02-2026\Fabian\spectrum\Spektren\Spektren"

# pick one file you know exists
TEST_FILE = "Output_RiKuLa_00000.csv"
path = os.path.join(SPECTRUM_FOLDER, TEST_FILE)

with open(path, "r", encoding="latin-1", errors="replace") as f:
    lines = f.readlines()

print("File:", path)
print("Total lines:", len(lines))
print("\n--- Showing first 120 lines (to see structure) ---")
for i, line in enumerate(lines[:120]):
    print(f"{i:03d}: {line.rstrip()}")
print("\n--- Done ---")


