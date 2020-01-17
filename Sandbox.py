import numpy as np

a = np.array([
    [1, 2, 2],
    [1, 3, 4],
    [2, 8, 9]
])

b = np.ndarray((3), dtype='uint8')
b.fill(5)

print(a)
print(b)

print(np.c_[a, b])
