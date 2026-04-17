![SpliCo Logo](assets/logo/logo.png)


The SpliCo library for spline-based geometric modeling.

The library's most important object is the `NDSplineArray` class, which
supports spline arithmetic using the default `NumPy` syntax.

For instance, let `disc` be a `NDSplineArray<npatches, 3>` that maps a
collection of `npatches` two-dimensional quadrilaterals onto a simply
connected two-dimensional manifold in R^3. Let `X` be a `NDSplineArray<3>`
that maps onto a one-dimensional manifold (curve) in R^3.

The syntax for sweeping the cross-sectional manifold `disc` along `X` is simply

```
_ = np.newaxis

vessel = (disc[:, _] @ rRs[_]).sum(-1) + disc.unity @ X
```

where `rRs` is a `NDSplineArray<3, 3>` representing (for instance) a
positionally-varying affine transformation of the cross-sectional
manifold.

In the above, we used standard `NumPy` broadcasting rules for matrix
multiplication and other operations.

The library supports:

- Spline arithmetic using the `NumPy` syntax and interoperability with 
`NumPy` arrays.
- Spline evaluation
- Spline fitting (in any dimension)
- Discrete twist-minimizing frame generation
- Sampling classical meshes of various types from the splines along with a
comprehensive module supporting various boolean mesh operations
- Various spline interpolation techniques (linear, cubic-Hermite, ...)
- The library provides an interface to the [Nutils](https://github.com/evalf/nutils/tree/main) finite element library
which supports variational techniques based on isogeometric analysis

## Installation

`pyhaemo` is a PEP 621 project built with [flit](https://flit.pypa.io/). From the
repository root (where `pyproject.toml` lives):

### Core library

```bash
pip install .
```

Pulls in the scientific stack used by the core package:
`numpy`, `scipy`, `nutils`, `treelog`, `numba`, `matplotlib`, `vtk`,
`pyvista`, `pygmsh`, `networkx`.

### Core library + GUI

The interactive mesh generator under `partery/` (launched by the scripts in
`scripts_GUI/`) additionally needs `PyQt6` and `jax`.  Install them together
with the core library through the **`gui`** extra:

```bash
pip install '.[gui]'
```

The convenience alias **`all`** currently pulls the same set:

```bash
pip install '.[all]'
```

### Editable / development install

```bash
pip install -e '.[gui]'
```

### Running the test suite

```bash
python -m unittest discover -s test -v
```

## GUI launcher install (Linux / macOS)

### Linux

```bash
./scripts/install_gui_launcher.sh
```

This writes:
- `~/.local/share/applications/pyhaemo-mesh-generator.desktop`

### macOS

```bash
./scripts/install_gui_launcher_macos.sh
```

This creates:
- `~/Applications/PyHaemo Mesh Generator.app`

Then launch **PyHaemo Mesh Generator** from the app menu/Applications and pin it to the dock.
