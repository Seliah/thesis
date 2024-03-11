"""Module for trying out the storage intensity of large numpy matrices."""
import rich
from numpy import zeros
from scipy.sparse import csr_array, random

DAY_IN_SECONDS = 24 * 60 * 60

rich.print("NumPy matrices")
nbytes = zeros((9, 16, 1000), dtype=bool).nbytes
rich.print(f"1000 timepoints: {nbytes} ({nbytes / 1_000_000} MB)")

nbytes = zeros((9, 16, 10000), dtype=bool).nbytes
rich.print(f"10000 timepoints: {nbytes} ({nbytes / 1_000_000} MB)")

nbytes = zeros((9, 16, DAY_IN_SECONDS), dtype=bool).nbytes
rich.print(f"{DAY_IN_SECONDS} timepoints (a day in seconds): {nbytes} ({nbytes / 1_000_000} MB)")
rich.print()

rich.print("SciPy matrix")
matrix = csr_array((144, DAY_IN_SECONDS), dtype=bool)
nbytes = matrix.data.nbytes + matrix.indptr.nbytes + matrix.indices.nbytes
print(f"{DAY_IN_SECONDS} timepoints (a day in seconds): {nbytes} ({nbytes / 1_000_000} MB)")

rich.print("SciPy matrix (randomized)")
matrix = random(144, DAY_IN_SECONDS, dtype=bool, density=0.5)
nbytes = matrix.data.nbytes
rich.print(f"{DAY_IN_SECONDS} timepoints (a day in seconds): {nbytes} ({nbytes / 1_000_000} MB)")

rich.print(matrix.toarray())
