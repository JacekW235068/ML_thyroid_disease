#!/usr/bin/env python3
import glob, os

all_paths = glob.glob("output/*/*/*.csv")
suffixes = list(set(map(lambda path : "_" + path.split("_")[-1], all_paths)))

os.makedirs("output_averaged", exist_ok=True)

for suffix in suffixes:
    fout = open("output_averaged/averaged" + suffix, "w")
    files = list(map(open, filter(lambda path : path.endswith(suffix), all_paths)))

    for i, line in enumerate(files[0]):
        lines = list(map(lambda file : file.readline(), files[1:]))
        if i == 0:
            fout.write(line)
            continue
        result_values = list(map(float, line.split(",")))
        for line in lines:
            values = list(map(float, line.split(",")))
            for i, value in enumerate(values):
                result_values[i] += value
        fout.write(",".join(map(lambda value : str(value / len(files)), result_values)) + "\n")

    for file in files:
        file.close()
    fout.close()
