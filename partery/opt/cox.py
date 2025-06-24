from pyhaemo.util import np
from pyhaemo.spl import UnivariateKnotVector


def _cox_de_boor_recursion(knotvector, funcindices, ksindices, coefficients, degree):
  """
  Docstring forthcoming.
  """
  # XXX: improve efficiency

  # { i: {j: coeff} }
  map_func_coeffs = {}

  # N_{i,p} = (x - t_i) / (t_{i + p} - t_i)           * N_{i, p-1} + \
  #           (t_{i+p+1} - x) / (t_{i+p+1} - t_{i+1}) * N_{i+1, p-1}
  # So N_{i, p-1} contributes (x - t_i) / (t_{i+p} - t_i) to N_{i, p}
  # let j = i+1.
  # Then N_{j, p-1} contributes to N_{j-1, p} the factor (t_{j+p} - x) / (t_{j+p} - t_j)
  for ifunc, ielem, coeffs in zip(funcindices, ksindices, coefficients):
    denom = knotvector[ifunc + degree] - knotvector[ifunc]
    if denom:
      elems = map_func_coeffs.setdefault(ifunc, {})
      mycoeffs = elems.setdefault(ielem, np.zeros(len(coeffs) + 1))
      mycoeffs += np.polymul( [1/denom, -knotvector[ifunc]/denom], coeffs[::-1] )[::-1]

      elems = map_func_coeffs.setdefault(ifunc-1, {})
      mycoeffs = elems.setdefault(ielem, np.zeros(len(coeffs) + 1))
      mycoeffs += np.polymul( [-1/denom, knotvector[ifunc+degree]/denom], coeffs[::-1] )[::-1]

  funcindices, elemindices, coeffs_per_element = [], [], []
  for ifunc, map_elem_coeffs in map_func_coeffs.items():
    for ielem, coeffs in map_elem_coeffs.items():
      funcindices.append(ifunc)
      elemindices.append(ielem)
      coeffs_per_element.append(coeffs)

  return funcindices, elemindices, coeffs_per_element


def univariate_spline_basis(knotvector: UnivariateKnotVector):
  knots = knotvector.repeated_knots

  elems0 = np.arange(len(knots)-knotvector.degree)
  funcindices, elemindices, coeffs = elems0, elems0, np.ones((len(elems0), 1), dtype=float)

  for i in range(knotvector.degree):
    funcindices, elemindices, coeffs = _cox_de_boor_recursion(knots,
                                                              funcindices,
                                                              elemindices,
                                                              coeffs,
                                                              i+1)
  _, elemindices = np.unique(elemindices, return_inverse=True)

  lexsort = np.lexsort((funcindices, elemindices))
  return np.array(funcindices, dtype=int)[lexsort], \
         np.array(elemindices, dtype=int)[lexsort], \
         np.stack(coeffs, axis=0)[lexsort]
