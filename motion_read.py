from datetime import datetime, timedelta
from pathlib import Path
from time import perf_counter

from numpy import bitwise_or, load
from scipy.sparse import csr_array

from util.time import TIME_ZERO

with Path("motions.npy").open("rb") as f:
    a = load(f, allow_pickle=True)
    # t = a[0][14][71]
    start_time = perf_counter()
    # print(a.nbytes)
    b = a[0][0]
    for y in range(a.shape[0]):
        for x in range(a.shape[1]):
            b = bitwise_or(b, a[y][x])
    for index, value in enumerate(b):
        if value:
            print(
                datetime.combine(datetime.now(), TIME_ZERO) + timedelta(seconds=index)
            )
            pass

    end_time = perf_counter()
    execution_time = end_time - start_time
    print(f"The execution time is: {execution_time}s")

with Path("motions.npy").open("rb") as f:
    a = load(f)

    start_time = perf_counter()
    # print(a.nbytes)
    # coo = coo_array(a)
    # coo.toarray().nbytes
    first_row = a[0][0]
    merged = csr_array(first_row, dtype=bool)
    for y in range(a.shape[0]):
        for x in range(a.shape[1]):
            row = csr_array(a[y][x], dtype=bool)
            merged = merged + row
    # merged[0, 2] = True
    print(merged)

    end_time = perf_counter()
    execution_time = end_time - start_time
    print(f"The execution time is: {execution_time}s")
