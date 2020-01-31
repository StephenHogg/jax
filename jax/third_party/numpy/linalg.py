from __future__ import absolute_import, division, print_function

import numpy as onp

from jax.numpy import lax_numpy as np
from jax.numpy import linalg as la
from jax.numpy.lax_numpy import _wraps
from jax.ops import index_update, index

def _isEmpty2d(arr):
  # check size first for efficiency
  return arr.size == 0 and np.product(arr.shape[-2:]) == 0


def _assertNoEmpty2d(*arrays):
  for a in arrays:
    if _isEmpty2d(a):
      raise onp.linalg.LinAlgError("Arrays cannot be empty")


def _assertRankAtLeast2(*arrays):
  for a in arrays:
    if a.ndim < 2:
      raise onp.linalg.LinAlgError(
          '%d-dimensional array given. Array must be '
          'at least two-dimensional' % a.ndim)


def _assertRank2(*arrays):
  for a in arrays:
    if a.ndim != 2:
      raise onp.linalg.LinAlgError('%d-dimensional array given. Array must be '
                                   'two-dimensional' % a.ndim)


def _assertNdSquareness(*arrays):
  for a in arrays:
    m, n = a.shape[-2:]
    if m != n:
      raise onp.linalg.LinAlgError(
          'Last 2 dimensions of the array must be square')


def _multi_dot(arrays, order, i, j):
  """Actually do the multiplication with the given order."""
  if i == j:
    return arrays[i]
  else:
    return np.dot(_multi_dot(arrays, order, i, order[i, j]),
                  _multi_dot(arrays, order, order[i, j] + 1, j))


def _multi_dot_three(A, B, C):
  """
  Find the best order for three arrays and do the multiplication.
  For three arguments `_multi_dot_three` is approximately 15 times faster
  than `_multi_dot_matrix_chain_order`
  """
  a0, a1b0 = A.shape
  b1c0, c1 = C.shape
  # cost1 = cost((AB)C) = a0*a1b0*b1c0 + a0*b1c0*c1
  cost1 = a0 * b1c0 * (a1b0 + c1)
  # cost2 = cost(A(BC)) = a1b0*b1c0*c1 + a0*a1b0*c1
  cost2 = a1b0 * c1 * (a0 + b1c0)

  if cost1 < cost2:
    return np.dot(np.dot(A, B), C)
  else:
    return np.dot(A, np.dot(B, C))


def _multi_dot_matrix_chain_order(arrays, return_costs=False):
  """
  Return a np.array that encodes the optimal order of mutiplications.
  The optimal order array is then used by `_multi_dot()` to do the
  multiplication.
  Also return the cost matrix if `return_costs` is `True`
  The implementation CLOSELY follows Cormen, "Introduction to Algorithms",
  Chapter 15.2, p. 370-378.  Note that Cormen uses 1-based indices.
      cost[i, j] = min([
          cost[prefix] + cost[suffix] + cost_mult(prefix, suffix)
          for k in range(i, j)])
  """
  n = len(arrays)
  # p stores the dimensions of the matrices
  # Example for p: A_{10x100}, B_{100x5}, C_{5x50} --> p = [10, 100, 5, 50]
  p = [a.shape[0] for a in arrays] + [arrays[-1].shape[1]]
  # m is a matrix of costs of the subproblems
  # m[i,j]: min number of scalar multiplications needed to compute A_{i..j}
  m = np.zeros((n, n), dtype=onp.double)
  # s is the actual ordering
  # s[i, j] is the value of k at which we split the product A_i..A_j
  s = np.empty((n, n), dtype=onp.intp)

  for l in range(1, n):
    for i in range(n - l):
      j = i + l
      index_update(m, index[i, j], onp.Inf)
      for k in range(i, j):
        q = m[i, k] + m[k+1, j] + p[i]*p[k+1]*p[j+1]
        if q < m[i, j]:
          index_update(m, index[i, j], q)
          index_update(s, index[i, j], k) # Note that Cormen uses 1-based index

  return (s, m) if return_costs else s


@_wraps(onp.linalg.cond)
def cond(a, p=None):
  x = np.asarray(a)  # in case we have a matrix
  _assertNoEmpty2d(x)
  if p is None or p == 2 or p == -2:
    s = la.svd(x, compute_uv=False)
    if p == -2:
      r = s[..., -1] / s[..., 0]
    else:
      r = s[..., 0] / s[..., -1]
  else:
    # Call inv(x) ignoring errors. The result array will
    # contain nans in the entries where inversion failed.
    _assertRankAtLeast2(x)
    _assertNdSquareness(x)
    invx = la.inv(x)
    r = la.norm(x, p, axis=(-2, -1)) * la.norm(invx, p, axis=(-2, -1))

  # Convert nans to infs unless the original array had nan entries
  r = np.asarray(r)
  nan_mask = np.isnan(r)
  if nan_mask.any():
    nan_mask &= ~np.isnan(x).any(axis=(-2, -1))
    if r.ndim > 0:
      r[nan_mask] = np.inf
    elif nan_mask:
      r[()] = np.inf

  # Convention is to return scalars instead of 0d arrays
  if r.ndim == 0:
    r = r[()]

  return r

@_wraps(onp.linalg.tensorinv)
def tensorinv(a, ind=2):
  a = np.asarray(a)
  oldshape = a.shape
  prod = 1
  if ind > 0:
    invshape = oldshape[ind:] + oldshape[:ind]
    for k in oldshape[ind:]:
      prod *= k
  else:
    raise ValueError("Invalid ind argument.")
  a = a.reshape(prod, -1)
  ia = la.inv(a)
  return ia.reshape(*invshape)

@_wraps(onp.linalg.multi_dot)
def multi_dot(arrays):
  n = len(arrays)
  # optimization only makes sense for len(arrays) > 2
  if n < 2:
    raise ValueError("Expecting at least two arrays.")
  elif n == 2:
    return np.dot(arrays[0], arrays[1])

  arrays = [np.asarray(a) for a in arrays]

  # save original ndim to reshape the result array into the proper form later
  ndim_first, ndim_last = arrays[0].ndim, arrays[-1].ndim
  # Explicitly convert vectors to 2D arrays to keep the logic of the internal
  # _multi_dot_* functions as simple as possible.
  if arrays[0].ndim == 1:
    arrays[0] = np.atleast_2d(arrays[0])
  if arrays[-1].ndim == 1:
    arrays[-1] = np.atleast_2d(arrays[-1]).T
  _assertRank2(*arrays)

  # _multi_dot_three is much faster than _multi_dot_matrix_chain_order
  if n == 3:
    result = _multi_dot_three(arrays[0], arrays[1], arrays[2])
  else:
    order = _multi_dot_matrix_chain_order(arrays)
    result = _multi_dot(arrays, order, 0, n - 1)

  # return proper shape
  if ndim_first == 1 and ndim_last == 1:
    return result[0, 0]  # scalar
  elif ndim_first == 1 or ndim_last == 1:
    return result.ravel()  # 1-D
  else:
    return result
