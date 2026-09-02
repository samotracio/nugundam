# Installation

νGundam contains compiled Fortran/OpenMP pair counters. For performance-sensitive work, the preferred installation is a **local source build** so the extension is compiled on the machine where it will run.

## Preferred: install 0.7.1 from source with pip

Install the build tools first.

=== "Ubuntu / Debian"

    ```bash
    sudo apt update
    sudo apt install build-essential gfortran cmake python3-dev
    ```

=== "Fedora / RHEL"

    ```bash
    sudo dnf install gcc gcc-gfortran cmake python3-devel
    ```

Create and activate a virtual environment, then tell pip not to use the νGundam wheel:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --no-binary=nugundam "nugundam==0.7.1"
```

`--no-binary=nugundam` applies only to νGundam. Binary wheels may still be used for NumPy, SciPy, Astropy, Matplotlib, and the build dependencies.

!!! important "Source build versus CPU-specific optimization"
    Building from source makes it possible to compile for the local toolchain and hardware, but the mere fact that pip compiled the package does not guarantee aggressive CPU-specific flags. For a machine-local build, optional compiler flags can be supplied explicitly:

    ```bash
    export CFLAGS="-O3 -march=native"
    export FFLAGS="-O3 -march=native"
    python -m pip install --no-binary=nugundam --no-cache-dir "nugundam==0.7.1"
    ```

    A binary produced with `-march=native` may not run on an older or different CPU. Use it only for an environment that will remain on compatible hardware.

## Wheel installation

A generic wheel is the simplest fallback:

```bash
python -m pip install "nugundam==0.7.1"
```

This is convenient for evaluation and portable environments, but a generic wheel cannot assume every instruction set available on a particular compute node.

## Install from a local source tree

From the repository root:

```bash
python -m pip install .
```

For editable development with the test dependencies:

```bash
python -m pip install -e ".[dev]"
pytest
```

A clean rebuild after changing compiler flags is safest:

```bash
python -m pip uninstall -y nugundam
rm -rf build _skbuild *.egg-info
python -m pip install --no-cache-dir .
```

## Verify the installation

νGundam 0.7.1 does not expose a top-level `__version__` attribute. Query the installed distribution metadata and compiled-backend status instead:

```python
from importlib.metadata import version

import nugundam
from nugundam.cflibfor import compiled_available

print(version("nugundam"))
print("compiled extension available:", compiled_available)
```

Expected output includes version `0.7.1` and `compiled extension available: True`.

!!! warning "Python import without the extension"
    The Python package can be imported when the compiled extension is absent so that configuration objects and documentation remain inspectable. Production pair counting requires the compiled backend. Always check `compiled_available` in a newly built environment.

## OpenMP and thread count

Set the number of threads in the νGundam configuration:

```python
cfg.nthreads = 16
```

`nthreads=-1` lets the OpenMP runtime choose. Cluster schedulers often also set `OMP_NUM_THREADS`; keep the scheduler allocation and the νGundam thread count consistent.

## Optional table backends

Astropy tables, NumPy structured arrays, and mappings are supported through the core dependencies. Install optional libraries only when those catalog types are used:

```bash
python -m pip install pandas pyarrow
```

## Build this documentation locally

From a repository containing `mkdocs.yml`, `docs/`, and `src/`:

```bash
python -m pip install -r docs/requirements.txt
mkdocs serve
```

Open `http://127.0.0.1:8000/`. The development server rebuilds the site after changes.

Run the same strict static build used for validation:

```bash
mkdocs build --strict
```

MathJax is loaded from a CDN, so equations render in a local browser while an internet connection is available.

## Inspect the configuration schema

Every configuration dataclass inherits a `describe()` helper:

```python
from nugundam import ProjectedAutoConfig

print(ProjectedAutoConfig.describe(recursive=True))
```

Resolved binning objects also provide human-readable tables:

```python
print(cfg.binning.table("rp"))
print(cfg.binning.table("pi"))
```
