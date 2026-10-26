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
            divisions = results[1]

            # everything in between is the polyarray string
            polyarrays_str = " ".join(parts[1:-3])

            try:
                polyarrays = ast.literal_eval(polyarrays_str)
            except Exception as e:
                print(f"[SKIP line {lineno}] parse error: {e}")
                continue

            # Reformat
            polys_fmt = []
            for poly in polyarrays:
                monoms = []
                for coef, degs in poly:
                    # degs is a list like [2,6,3] → turn into tuple
                    monoms.append(f"({coef},({','.join(map(str,degs))}))")
                polys_fmt.append(",".join(monoms))
            polys_str = ";".join(polys_fmt)

            out_line = f"({ordering})|{polys_str}|{divisions}\n"
            f_out.write(out_line)

            print(f"[OK line {lineno}] {out_line.strip()}")

# Example usage
reformat_dataset(
    infile=r"C:\Users\HP\PycharmProjects\PythonProject4\data100.txt",
    outfile="data100reformatted.txt"
)
