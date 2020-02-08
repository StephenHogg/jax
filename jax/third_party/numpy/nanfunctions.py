import numpy as onp

from jax.numpy.lax_numpy import _wraps

import jax.numpy as np


@_wraps(onp.nanargmin)
def nanargmin(a, axis=None):
    mask = a.isnan()
    a, mask = np.where(mask, np.inf, a)
    res = np.argmin(a, axis=axis)
    if mask is not None:
        mask = np.all(mask, axis=axis)
        if np.any(mask):
            raise ValueError("All-NaN slice encountered")
    return res


@_wraps(onp.nanargmax)
def nanargmax(a, axis=None):
    mask = a.isnan()
    a = np.where(mask, -np.inf, a)
    res = np.argmax(a, axis=axis)
    if mask is not None:
        mask = np.all(mask, axis=axis)
        if np.any(mask):
            raise ValueError("All-NaN slice encountered")
    return res


@_wraps(onp.nanstd)
def nanstd(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    if out is not None:
        msg = ("jax.numpy.nanstd does not support out != None")
        raise ValueError(msg)
    var = nanvar(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
                 keepdims=keepdims)
    if isinstance(var, np.ndarray):
        std = np.sqrt(var)
    else:
        std = var.dtype.type(np.sqrt(var))
    return std


@_wraps(onp.nanvar)
def nanvar(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    if out is not None:
        msg = ("jax.numpy.nanvar does not support out != None")
        raise ValueError(msg)
    mask = np.isnan(a)
    arr = np.where(mask, 0, a)
    if not mask.any():
        return np.var(arr, axis=axis, dtype=dtype, out=out, ddof=ddof,
                      keepdims=keepdims)

    if dtype is not None:
        dtype = np.dtype(dtype)
    if dtype is not None and not issubclass(dtype.type, np.inexact):
        raise TypeError("If a is inexact, then dtype must be inexact")
    if out is not None and not issubclass(out.dtype.type, np.inexact):
        raise TypeError("If a is inexact, then out must be inexact")

    # Compute mean
    cnt = np.sum(~mask, axis=axis, dtype=np.int8, keepdims=keepdims)
    avg = np.sum(arr, axis=axis, dtype=dtype, keepdims=keepdims)
    avg = np.divide(avg, cnt)

    # Compute squared deviation from mean.
    dev = onp.subtract(arr, avg)
    dev = np.where(mask, np.nan, dev)
    if issubclass(arr.dtype.type, np.complexfloating):
        sqr = np.multiply(dev, dev.conj()).real
    else:
        sqr = np.multiply(dev, dev)

    # Compute variance.
    var = np.sum(sqr, axis=axis, dtype=dtype, keepdims=keepdims)
    if var.ndim < cnt.ndim:
        # Subclasses of ndarray may ignore keepdims, so check here.
        cnt = cnt.squeeze(axis)
    dof = cnt - ddof
    var = np.divide(var, dof)

    isbad = (dof <= 0)
    var = np.where(isbad, np.nan, var)
    return var
