import os
import difflib
import random

def generate(n, m, nnz, filename):
    os.system(f"python3 generate.py {n} {m} {nnz} > {filename}")

def main(A, B, pc, l, mode, output):
    os.system(f"mpiexec -n {pc * pc * l} ./main -a {A} -b {B} -t {mode} -l {l} -v > {output}")

def brute(A, B, output):
    os.system(f"./brute {A} {B} > {output}")

def read_file(filename):
    with open(filename, 'r') as file:
        return file.readlines()

def diff(main_out, brute_out):
    diff = difflib.unified_diff(
        read_file(main_out), read_file(brute_out),
        fromfile=main_out, tofile=brute_out,
        lineterm=''
    )

    return bool([line for line in diff])

for i in range(100):
    print(f'running test number {i}')

    l = random.randint(1, 3)
    pc = random.randint(1, 3)
    n = random.randint(l * pc, 30)

    generate(n, n, random.randint(0, n * n), 'atest')
    generate(n, n, random.randint(0, n * n), 'btest')
    main('atest', 'btest', pc, l, '3D', 'main.out')
    brute('atest', 'btest', 'brute.out')

    if diff('main.out', 'brute.out'):
        print('diff failed')
        exit(1)

