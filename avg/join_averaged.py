#!/usr/bin/env python3
import glob

fout = open("output_averaged_joined.csv", "w")
for i, path in enumerate(glob.glob("output_averaged/*.csv")):
    file = open(path, "r")
    first_line = file.readline()
    if i == 0:
        fout.write(first_line)
    for line in file:
        fout.write(line)
    file.close()
fout.close()
