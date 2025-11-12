
import numpy as np


def to_oneD_array(
        array_like, 
        dtype=np.float32
        )->np.ndarray:
    """
    converts x to 1D array and if not possible, raises ValueError
    Returned array will have shape (n,)
    """
    if array_like.__class__.__name__ in ['list', 'tuple', 'Series', 'int', 'float']:
        return np.array(array_like, dtype=dtype)

    elif array_like.__class__.__name__ == 'ndarray':
        if array_like.ndim == 1:
            return array_like.astype(dtype)
        else:
            if array_like.size != len(array_like):
                raise ValueError(f'cannot convert multidim array of shape {array_like.shape} to 1d')

            return array_like.reshape(-1, ).astype(dtype)

    elif array_like.__class__.__name__ == 'DataFrame' and array_like.ndim == 2:
        assert len(array_like) == array_like.size
        return array_like.values.reshape(-1, ).astype(dtype)

    elif array_like.__class__.__name__ == "Series":
        return array_like.values.reshape(-1, ).astype(dtype)

    elif isinstance(array_like, float) or isinstance(array_like, int):
        return np.array([array_like]).astype(dtype)
    else:
        raise ValueError(f'cannot convert object {array_like.__class__.__name__}  to 1d ')
