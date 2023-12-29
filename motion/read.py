from datetime import datetime, timedelta
from functools import reduce
from operator import add
from pathlib import Path
from time import perf_counter

from numpy import load
from scipy.sparse import lil_array

from util.time import TIME_ZERO

with Path("motions.npy").open("rb") as f:
    a = load(f, allow_pickle=True)
    start_time = perf_counter()
    cam: lil_array = a[4][69]
    cells = [cam.getrow(cell_index) for cell_index in range(cam.shape[0])]
    merged = reduce(add, cells)
    for index in merged.nonzero()[1]:
        print(
            datetime.combine(datetime.now(), TIME_ZERO) + timedelta(seconds=int(index))
        )
    end_time = perf_counter()
    execution_time = end_time - start_time
    print(f"The execution time is: {execution_time}s")
