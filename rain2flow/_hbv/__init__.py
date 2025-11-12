
from ._main import hbv

try:
    import numba
except (ImportError, ModuleNotFoundError):
    numba = None

if numba is not None:
    from ._numba import hbv_nb
else:
    hbv_nb = None