import ast

def reformat_line(line: str) -> str:
    line = line.strip()
    if not line:
        return ""

    # Split into order, ideal_string, label
    first_space = line.find(" ")
    last_space = line.rfind(" ")

    order = line[:first_space]
    ideal_str = line[first_space + 1:last_space]
    label = line[last_space + 1:]

    try:
        ideal = ast.literal_eval(ideal_str)
    except Exception as e:
        print(f"Parse error on line:\n{line}\nError: {e}\n")
        return ""

    # Reformat each polynomial
    poly_strs = []
    for poly in ideal:
        monos = [f"({coeff},({','.join(map(str, exps))}))" for coeff, exps in poly]
        poly_strs.append(",".join(monos))

    # Join polynomials with ;
    ideal_formatted = ";".join(poly_strs)

    return f"({order})|{ideal_formatted}|{label}"


# === Main script ===
input_file = r"../data/dataset_222"
output_file = r"../data/data_formatted222.txt"

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        formatted = reformat_line(line)
        if formatted:
            outfile.write(formatted + "\n")

print(f"Reformatted dataset written to {output_file}")
