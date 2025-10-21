import pathway as pw

# Define the schema for the CSV file
class Measurement(pw.Schema):
    name: str
    value: float

# ✅ Read CSV file (use .read for modern Pathway)
measurements = pw.io.csv.read("data/measurements.csv", schema=Measurement)

# Filter positive values only
positive = measurements.filter(pw.this.value > 0)

# Select required columns
result = positive.select(pw.this.name, pw.this.value)

# ✅ Write results to a JSON Lines file
pw.io.jsonlines.write(result, "out/results.jsonl")

print("✅ Starting Pathway pipeline...")
pw.run()
