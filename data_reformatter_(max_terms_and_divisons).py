import ast

def reformat_dataset(infile, outfile):
    with open(infile, "r") as f_in, open(outfile, "w") as f_out:
        for lineno, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                print(f"[SKIP line {lineno}] not enough parts: {line}")
                continue

            ordering = parts[0]
            # last 3 = max_terms, divisions, time
            results = parts[-3:]
            max_terms = results[0]
            divisions = results[1]

            # everything in between is the polyarray string
            polyarrays_str = " ".join(parts[1:-3])

            try:
                polyarrays = ast.literal_eval(polyarrays_str)
            except Exception as e:
                print(f"[SKIP line {lineno}] parse error: {e}")
                continue

            # Reformat polys
            polys_fmt = []
            for poly in polyarrays:
                monoms = []
                for coef, degs in poly:
                    monoms.append(f"({coef},({','.join(map(str,degs))}))")
                polys_fmt.append(",".join(monoms))
            polys_str = ";".join(polys_fmt)

            # Write max_terms and divisions separately
            out_line = f"({ordering})|{polys_str}|{max_terms}|{divisions}\n"
            f_out.write(out_line)

            print(f"[OK line {lineno}] {out_line.strip()}")

# Example usage
reformat_dataset(
    infile=r"C:\Users\HP\PycharmProjects\PythonProject4\models\divions_data_with_-1",
    outfile="divisions_with_two_features_formatted.txt"
)