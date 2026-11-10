#!/usr/bin/env bash

# Run from the main folder, e.g. au-100ps/
# It automatically finds numeric folders: 1 2 3 ...
# Or you can give folders manually: ./process_md_runs.sh 1 2 3

set -u
ROOT="$(pwd)"
SCRIPT_DIR="${SCRIPT_DIR:-$ROOT}"

TEMP_SCRIPT="$SCRIPT_DIR/extract-temp-energy.py"
XDAT2XYZ_SCRIPT="$SCRIPT_DIR/xdat2xyz.pl"
XDAT_COMBINE_SCRIPT="$SCRIPT_DIR/xdat-combine.py"
COMBINED_MOVIE_SCRIPT="$SCRIPT_DIR/xdatcartoxyz-combinedmovie.py"

LOG_DIR="$ROOT/postprocess-logs"
mkdir -p "$LOG_DIR"

# Change these only if your scripts produce different file types
TEMP_PATTERN="*.csv"
XYZ_PATTERN="*.xyz"

# Set to 0 if you do not want to run the two main-folder combine scripts
RUN_MAIN_COMBINE="${RUN_MAIN_COMBINE:-1}"

check_script() {
    local script="$1"
    if [[ ! -f "$script" ]]; then
        echo "ERROR: Cannot find $script"
        exit 1
    fi
}

newest_file_after_marker() {
    local folder="$1"
    local marker="$2"
    local pattern="$3"

    find "$folder" -maxdepth 1 -type f -name "$pattern" -newer "$marker" \
        -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-
}

run_dirs=()

if [[ "$#" -gt 0 ]]; then
    run_dirs=("$@")
else
    while IFS= read -r d; do
        run_dirs+=("$d")
    done < <(find "$ROOT" -maxdepth 1 -mindepth 1 -type d -printf "%f\n" | grep -E '^[0-9]+$' | sort -V)
fi

if [[ "${#run_dirs[@]}" -eq 0 ]]; then
    echo "ERROR: No numeric folders found."
    exit 1
fi

check_script "$TEMP_SCRIPT"
check_script "$XDAT2XYZ_SCRIPT"

echo "Main folder: $ROOT"
echo "Folders to process: ${run_dirs[*]}"
echo

failed=()

for d in "${run_dirs[@]}"; do
    folder="$ROOT/$d"

    if [[ ! -d "$folder" ]]; then
        echo "Skipping $d: folder does not exist."
        failed+=("$d")
        continue
    fi

    echo "Processing folder $d ..."

    marker_temp="$LOG_DIR/marker-temp-$d"
    touch "$marker_temp"

    (
        cd "$folder" || exit 1
        python "$TEMP_SCRIPT"
    ) > "$LOG_DIR/extract-temp-energy-$d.log" 2>&1

    temp_status=$?

    if [[ "$temp_status" -ne 0 ]]; then
        echo "  WARNING: extract-temp-energy.py failed in folder $d. See postprocess-logs/extract-temp-energy-$d.log"
        failed+=("$d-temp")
    else
        temp_out="$(newest_file_after_marker "$folder" "$marker_temp" "$TEMP_PATTERN")"

        if [[ -n "$temp_out" ]]; then
            cp "$temp_out" "$ROOT/file-$d.csv"
            echo "  Saved: file-$d.csv"
        else
            echo "  WARNING: No CSV found after extract-temp-energy.py in folder $d"
            failed+=("$d-temp-output")
        fi
    fi

    marker_xyz="$LOG_DIR/marker-xyz-$d"
    touch "$marker_xyz"

    (
        cd "$folder" || exit 1
        perl "$XDAT2XYZ_SCRIPT"
    ) > "$LOG_DIR/xdat2xyz-$d.log" 2>&1

    xyz_status=$?

    if [[ "$xyz_status" -ne 0 ]]; then
        echo "  WARNING: xdat2xyz.pl failed in folder $d. See postprocess-logs/xdat2xyz-$d.log"
        failed+=("$d-xyz")
    else
        xyz_out="$(newest_file_after_marker "$folder" "$marker_xyz" "$XYZ_PATTERN")"

        if [[ -n "$xyz_out" ]]; then
            cp "$xyz_out" "$ROOT/movie-file-$d.xyz"
            echo "  Saved: movie-file-$d.xyz"
        else
            echo "  WARNING: No XYZ found after xdat2xyz.pl in folder $d"
            failed+=("$d-xyz-output")
        fi
    fi

    echo
done

if [[ "$RUN_MAIN_COMBINE" == "1" ]]; then
    echo "Running main-folder combine scripts ..."

    if [[ -f "$XDAT_COMBINE_SCRIPT" ]]; then
        (
            cd "$ROOT" || exit 1
            python "$XDAT_COMBINE_SCRIPT"
        ) > "$LOG_DIR/xdat-combine.log" 2>&1

        if [[ "$?" -eq 0 ]]; then
            echo "  Finished: xdat-combine.py"
        else
            echo "  WARNING: xdat-combine.py failed. See postprocess-logs/xdat-combine.log"
            failed+=("xdat-combine")
        fi
    else
        echo "  Skipping xdat-combine.py: not found."
    fi

    if [[ -f "$COMBINED_MOVIE_SCRIPT" ]]; then
        (
            cd "$ROOT" || exit 1
            python "$COMBINED_MOVIE_SCRIPT"
        ) > "$LOG_DIR/xdatcartoxyz-combinedmovie.log" 2>&1

        if [[ "$?" -eq 0 ]]; then
            echo "  Finished: xdatcartoxyz-combinedmovie.py"
        else
            echo "  WARNING: xdatcartoxyz-combinedmovie.py failed. See postprocess-logs/xdatcartoxyz-combinedmovie.log"
            failed+=("combinedmovie")
        fi
    else
        echo "  Skipping xdatcartoxyz-combinedmovie.py: not found."
    fi
fi

echo
echo "Done."

if [[ "${#failed[@]}" -gt 0 ]]; then
    echo "Some steps had warnings/failures:"
    printf '  %s\n' "${failed[@]}"
    echo "Check logs in: postprocess-logs/"
else
    echo "All folders processed successfully."
fi
