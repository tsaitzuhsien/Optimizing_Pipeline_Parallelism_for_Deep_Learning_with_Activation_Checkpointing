import sys

d1 = []
with open(sys.argv[1], 'r') as f:
    for i, m in enumerate(f):
        d1.append(float(m))

d2 = []
with open(sys.argv[2], 'r') as f:
    for i, m in enumerate(f):
        d2.append(float(m))

rr = []
for i, j in zip(d1,d2):
    rr.append(abs(i - j) / i)

print(sum(rr) / len(rr))
