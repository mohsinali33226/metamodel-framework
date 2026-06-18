from pathlib import Path


def read_spectrum_root() -> Path:
    # Project root = the folder that contains "data" and "src"
    project_root = Path(__file__).resolve().parents[1]
    path_file = project_root / "data" / "raw" / "spectrum_path.txt"
    spectrum_root_str = path_file.read_text(encoding="utf-8").strip()
    return Path(spectrum_root_str)


def find_header_line(lines: list[str]) -> int | None:
    # We search the line that contains BOTH "time [s]" and "f[Hz]"
    for i, line in enumerate(lines):
        s = line.strip().lower()
        if ("time [s]" in s) and ("f[hz]" in s):
            return i
    return None


def split_header_to_names(header_line: str) -> list[str]:
    # Files are semicolon-separated (;) in your screenshots
    raw_parts = [p.strip() for p in header_line.split(";")]
    # Remove empty items and normalize spaces
    parts = [p for p in raw_parts if p != ""]
    # Optional: simplify multiple spaces
    parts = [" ".join(p.split()) for p in parts]
    return parts


def main():
    spectrum_root = read_spectrum_root()
    files = sorted(spectrum_root.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {spectrum_root}")

    example = files[0]
    lines = example.read_text(encoding="utf-8", errors="ignore").splitlines()

    idx = find_header_line(lines)
    print("Example file:", example.name)
    print("Header line index (0-based):", idx)

    if idx is None:
        print("Could not find the combined header line containing time[s] and f[Hz].")
        return

    header_line = lines[idx].strip()
    names = split_header_to_names(header_line)

    print("\n--- Header line (raw) ---")
    print(header_line)

    print("\n--- Parsed column names ---")
    for k, name in enumerate(names):
        print(f"{k:02d}: {name}")

    print("\nNumber of parsed columns:", len(names))


if __name__ == "__main__":
    main()


