import re
import hashlib
import random

def deterministic_shuffle(polys, seed_str):
    """Shuffles deterministically using a hash of the seed string."""
    seed = int(hashlib.sha256(seed_str.encode('utf-8')).hexdigest(), 16)
    rng = random.Random(seed)

    # Shuffle until it changes
    original = polys.copy()
    if len(polys) > 1:
        for _ in range(10):  # limit attempts
            shuffled = polys.copy()
            rng.shuffle(shuffled)
            if shuffled != original:
                return shuffled
    return polys  # fallback, should almost never happen


def randomize_polynomials(line):
    # Match format: (3)|poly1;poly2;...;polyN|42
    match = re.match(r'^\((\d+)\)\|(.*)\|(\d+)$', line.strip())
    if not match:
        return None  # skip invalid lines

    order, body, s_poly_count = match.groups()
    polys = [p.strip() for p in body.split(';') if p.strip()]

    # Skip single-polynomial lines
    if len(polys) <= 1:
        return None

    # Deterministically shuffle
    polys = deterministic_shuffle(polys, line)

    return f"({order})|{';'.join(polys)}|{s_poly_count}"


def process_file(input_file, output_file):
    kept, removed = 0, 0
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            new_line = randomize_polynomials(line)
            if new_line:
                fout.write(new_line + "\n")
                kept += 1
            else:
                removed += 1

    print(f"✅ Wrote {kept} lines to {output_file}")
    print(f"🚫 Removed {removed} single-poly or malformed lines")


# Example usage
if __name__ == "__main__":
    input_path = r"../data/bigonewith100_no-1.txt"
    output_path = r"../data/bigonewith100_no-1.txt"
    process_file(input_path, output_path)
