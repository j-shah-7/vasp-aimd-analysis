#!/usr/bin/env python3
import sys
import os

# Default paths match xdat-combine.py
input_xdatcar = "combined_xdatcar/XDATCAR_combined"
output_xyz = "movie-combined.xyz"

# Usage:
#   python xdatcartoxyz-combinedmovie.py
#   python xdatcartoxyz-combinedmovie.py combined_xdatcar/XDATCAR_combined movie-combined.xyz

if len(sys.argv) > 1:
    input_xdatcar = sys.argv[1]

if len(sys.argv) > 2:
    output_xyz = sys.argv[2]


def is_config_line(line):
    return line.strip().startswith("Direct configuration=") or line.strip().startswith("Cartesian configuration=")


def read_header_and_first_config(f):
    header = []

    for line in f:
        if is_config_line(line):
            return header, line

        header.append(line)

    raise ValueError("No Direct/Cartesian configuration lines found.")


def direct_to_cart(frac, lattice):
    x, y, z = frac

    cart_x = x * lattice[0][0] + y * lattice[1][0] + z * lattice[2][0]
    cart_y = x * lattice[0][1] + y * lattice[1][1] + z * lattice[2][1]
    cart_z = x * lattice[0][2] + y * lattice[1][2] + z * lattice[2][2]

    return cart_x, cart_y, cart_z


def frame_iterator(f, natoms, first_config_line):
    config_line = first_config_line

    while config_line is not None:
        frame = []

        for _ in range(natoms):
            line = f.readline()

            if not line:
                print("Warning: incomplete final frame; skipping.")
                return

            parts = line.split()

            if len(parts) < 3:
                print("Warning: bad coordinate line; skipping incomplete frame.")
                return

            frame.append([float(parts[0]), float(parts[1]), float(parts[2])])

        yield frame

        config_line = None

        for line in f:
            if is_config_line(line):
                config_line = line
                break
            elif line.strip() == "":
                continue
            else:
                continue


if not os.path.isfile(input_xdatcar):
    print(f"Error: cannot find {input_xdatcar}")
    sys.exit(1)

with open(input_xdatcar, "r", errors="ignore") as f:
    header, first_config_line = read_header_and_first_config(f)

    if len(header) < 7:
        raise ValueError("XDATCAR header is too short.")

    scale = float(header[1].split()[0])

    lattice = []
    for i in range(2, 5):
        lattice.append([float(x) * scale for x in header[i].split()[:3]])

    elements = header[5].split()
    counts = [int(x) for x in header[6].split()]

    atom_types = []
    for element, count in zip(elements, counts):
        atom_types.extend([element] * count)

    natoms = sum(counts)

    frame_count = 0

    with open(output_xyz, "w") as out:
        for frame in frame_iterator(f, natoms, first_config_line):
            frame_count += 1

            out.write(f"{natoms}\n")
            out.write(f"Frame {frame_count}\n")

            for atom, frac in zip(atom_types, frame):
                x, y, z = direct_to_cart(frac, lattice)
                out.write(f"{atom:2s} {x:18.10f} {y:18.10f} {z:18.10f}\n")

print(f"Read {frame_count} frames from {input_xdatcar}")
print(f"Wrote {output_xyz}")
print(f"Atoms per frame: {natoms}")
