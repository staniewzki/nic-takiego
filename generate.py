import sys
import random

if len(sys.argv) != 4:
    print(f"usage: {sys.argv[0]} n m nnz")
    exit(1)

n = int(sys.argv[1])
m = int(sys.argv[2])
nnz = int(sys.argv[3])

if n * m < nnz:
    print("error: too much non-zero values");
    exit(1)

print(f'{n} {m} {nnz} {m}')

for i in range(nnz):
    print(random.randint(-2, 2), end=' ')
print()

rows = [0] * n
added = 0

while added < nnz:
    i = random.randint(0, n - 1)
    if rows[i] != m:
        rows[i] += 1
        added += 1

rows = [0] + rows
for i in range(1, n + 1):
    rows[i] = rows[i] + rows[i - 1]

for i in range(n):
    cnt = rows[i + 1] - rows[i]
    sample = random.sample(range(0, m), cnt)
    sample.sort()
    for j in sample:
        print(j, end=' ')
print()

for x in rows:
    print(x, end=' ')
print()

