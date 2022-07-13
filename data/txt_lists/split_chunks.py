import sys
import os 
from tqdm import tqdm

assert len(sys.argv) == 4, "Launch with params: input_file, num_of_chunks, output_dir"

input_file = sys.argv[1]
n_chunks = int(sys.argv[2])
output_dir = sys.argv[3]

assert os.path.isfile(input_file), f"{input_file} does not exist"
assert os.path.isdir(output_dir), f"Output dir {output_dir} does not exist"

basename = os.path.basename(input_file).split(".")[0]

with open(input_file, "r") as inf:
    lines = inf.readlines()

out_files = []
for idx in range(n_chunks):
    out_files.append(open(os.path.join(output_dir, f"{basename}_{idx}.txt"), "w"))

for l_idx, l in enumerate(tqdm(lines)):
    chunk_id = l_idx % n_chunks
    out_files[chunk_id].write(l)

for idx in range(n_chunks):
    print(f"{os.path.join(output_dir, f'{basename}_{idx}.txt')} written!")
    out_files[idx].close()



