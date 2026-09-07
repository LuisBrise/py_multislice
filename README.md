[![DOI](https://zenodo.org/badge/209026254.svg)](https://zenodo.org/badge/latestdoi/209026254)
# py_multislice

![](cbed.png)

Python multislice slice code

GPU accelerated using 
[pytorch](https://pytorch.org/)

Ionization based off [Flexible Atomic Code (FAC)](https://github.com/flexible-atomic-code/fac).

**This branch (`magnetic-phase-modulation`)** extends `py_multislice` with magnetic
phase modulation: an Aharonov-Bohm phase shift from a specimen's in-plane atomic
magnetic moments, added on top of the usual electrostatic transmission function.
See [Magnetic scattering](#magnetic-scattering) below.

# Installation

1. Clone or branch this repo into a directory on your computer

```bash
    $ git clone https://github.com/HamishGBrown/py_multislice.git
```

2. (Optional) create a new conda environment for py_multislice:

```bash
    $ conda create --name py_multislice
    $ conda activate py_multislice
```

3. In the command-line (Linux or Mac) or your Python interpreter (Windows) install pytorch, you will need to choose the conda version appropriate for your GPU (or CPU only version if required) see [here](https://pytorch.org/get-started/locally/), and in the root directory of your local copy of the repo run the install command for py_multislice

```bash
    $ conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
    $ pip install -e .
```

   All necessary dependencies should also be installed, if you have issues try installing in a fresh anaconda environment (step 2).

4. If you would like to perform ionization based TEM simulations, download and install the flexible atomic code (FAC), including the python interface (pfac), from [here](https://github.com/flexible-atomic-code/fac). I've only successfully got this working on Linux, your mileage may vary on Windows operating systems. 

5. As an added precaution, run the Test.py script to ensure everything is working as expected

```bash
    $ python Test.py
```

    If you didn't instal PFAC in the last step then you will get error messages from the ionization routines, if you only want to perform non-ionization based TEM simulations then you can ignore these failed tests. You can also run run Orbital_normalization.py to test that the ionization cross-sections are being calculated appropriately.

# Documentation and demos

Documentation can be found [here](https://hamishgbrown.github.io/py_multislice/pyms/), for demonstrations and walk throughs on common simulation types see the Jupyter Notebooks in the [Demos](Demos/) folder. The Notebook for STEM-EELS is still under construction.

# Magnetic scattering

This branch adds magnetic phase modulation to `pyms.structure_routines.structure`:
an Aharonov-Bohm phase shift, `exp[-i (e/hbar) integral(A_z dz)]`, imprinted by
the in-plane component of a specimen's atomic magnetic moments, computed
alongside (and added into) the ordinary electrostatic transmission function.
Only the in-plane moment components contribute to this phase -- a moment
purely along the beam direction contributes exactly zero, a standard result
for collinear magnetism (see Edstrom, Lubk & Rusz, *Magnetic effects in the
paraxial regime of elastic electron scattering*, Phys. Rev. B 94, 174414
(2016)). This is one specific channel of the full magnetic Pauli-equation
Hamiltonian; see the docstrings in `pyms/structure_routines.py` for the full
scope, references, and design notes.

**What's new:**

- `structure(unitcell, atoms, dwf, magnetic_moments=...)` -- an optional,
  default-`None` `(natoms, 3)` array of Cartesian (mx, my, mz) moments per
  atom, in Bohr magnetons.
- `structure.make_magnetic_potential(...)` -- the magnetic phase on its own,
  mirroring the existing `make_potential`'s call signature.
- `structure.make_transmission_functions(..., include_magnetic=True)` -- adds
  the magnetic phase into the combined transmission function (the default).
  A structure with no magnetic moment data (the default, `magnetic_moments=
  None`) is completely unaffected by this flag -- every pre-existing,
  non-magnetic call site keeps working exactly as before.

**Examples:**

- [`Demos/Magnetic_Phase_Modulation.ipynb`](Demos/Magnetic_Phase_Modulation.ipynb) --
  a worked walkthrough: building a magnetic structure, the magnetic phase
  alongside the electrostatic potential, the combined transmission function,
  and the (small but real) effect on a propagated exit wave and diffraction
  pattern.
- `tests/test_magnetic_form_factors.py`, `tests/test_make_magnetic_potential.py`,
  `tests/test_make_transmission_functions_magnetic.py` -- 35 focused unit
  tests that double as runnable usage examples, including zero-net-moment
  structures, rotational equivariance checks, and backward-compatibility
  checks against non-magnetic structures.

# Bug-fixes and contributions

Message me, leave a bug report and fork the repo. All contributions are welcome.

# Acknowledgements

A big thanks to [Philipp Pelz](https://github.com/PhilippPelz) for teaching me the ins and outs of pytorch and numerous other discussions on computing and electron microscopy. Credit to [Colin Ophus](https://github.com/cophus) for many discussions and much inspiration re the PRISM algorithm (of which he is the inventor). Thanks to my boss Jim Ciston for tolerating this side project! Thankyou to [Thomas Aarholt](https://github.com/thomasaarholt) for Python advice and testing of different libraries.  Thanks to Adrian D'Alfonso, Scott Findlay and Les Allen (my PhD advisor) for originally teaching me the art of multislice and ionization.


