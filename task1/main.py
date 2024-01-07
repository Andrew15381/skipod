def write(qw, ts, tb):
    a = qw * (ts + tb * 4)
    b = qw * (ts + tb * 4)
    c = qw * (ts + tb * 3000)
    return a + b + c

def read(qr, ts, tb):
    a = qr * (ts + tb * 4)
    b = qr * (ts + tb * 4)
    c = ts + tb * 3000
    return a + b + c

def t(qw, qr, ts, tb):
    return write(qw, ts, tb) * 4 + read(qr, ts, tb) * 10

a = []
for i in range(1, 21):
    for j in range(1, 21):
            if i + j <= 20:
                continue
            a.append([i, j, t(i, j, 100, 1)])

print(sorted(a, key=lambda x: x[2]))
