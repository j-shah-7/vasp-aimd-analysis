import re

input_filename = "stdout"   # change if your file name differs
output_csv = "temperature_energy.csv"
output_dat = "temperature_energy.dat"

pattern = re.compile(
    r"^\s*(\d+)\s+T=\s*([-0-9.]+)\.\s+E=\s*([-0-9.Ee+]+)"
)

data = []

with open(input_filename, "r", errors="ignore") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            step = int(match.group(1))
            temperature = float(match.group(2))
            energy = float(match.group(3))
            data.append((step, temperature, energy))

if not data:
    print("No matching lines found.")
    exit()

# Write CSV
with open(output_csv, "w") as f:
    f.write("step,temperature_K,energy_eV\n")
    for step, T, E in data:
        f.write(f"{step},{T},{E}\n")

# Write DAT
with open(output_dat, "w") as f:
    f.write("# step temperature(K) energy(eV)\n")
    for step, T, E in data:
        f.write(f"{step} {T} {E}\n")

print(f"Extracted {len(data)} steps.")
print(f"Wrote: {output_csv} and {output_dat}")

