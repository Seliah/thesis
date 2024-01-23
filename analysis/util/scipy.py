"""Module for expanding or typing scipy functionality."""
from __future__ import annotations

from functools import reduce
from operator import add
from typing import cast

from scipy.sparse import lil_array


def nonzero(matrix: lil_array) -> tuple[list[int], list[int]]:
    """scipy.sparse.lil_array.nonzero but typed.

    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_array.nonzero.html#scipy.sparse.lil_array.nonzero
    """
    return cast(tuple[list[int], list[int]], matrix.nonzero())


def getrow(matrix: lil_array, i: int):
    """scipy.sparse.lil_array.getrow but typed.

    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_array.getrow.html#scipy.sparse.lil_array.getrow
    """
    return cast(lil_array, matrix.getrow(i))  # pyright: ignore[reportUnknownMemberType]


def nnz(matrix: lil_array):
    """scipy.sparse.lil_array.nnz but typed.

    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_array.nnz.html#scipy.sparse.lil_array.nnz
    """
    return cast(int, matrix.nnz)  # pyright: ignore[reportUnknownMemberType]


def combine_or(matrices: list[lil_array]) -> lil_array:
    """Combine all given matrices with or (bitwise)."""
    return reduce(add, matrices)
