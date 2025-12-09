import glob

def merge_files(pattern, output_file):
    files = sorted(glob.glob(pattern))
    with open(output_file, 'wb') as outfile:
        for f in files:
            with open(f, 'rb') as infile:
                outfile.write(infile.read())
    print(f"Merged {len(files)} files into {output_file}")

if __name__ == "__main__":
    merge_files("linear_rank*.dat", "linear_combined.dat")
    merge_files("boost_rank*.dat", "boost_combined.dat")

