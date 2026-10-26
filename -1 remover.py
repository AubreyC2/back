input_file = r"/big_one_reformatted"  # your original file
output_file = "../data/big_one_no_-1"  # new file without -1 lines

with open(input_file, "r") as fin, open(output_file, "w") as fout:
    for line in fin:
        parts = line.strip().split("|")
        if len(parts) >= 3:
            last_value = parts[2].strip()
            if last_value == "-1":  # skip if -1
                continue
        fout.write(line)
