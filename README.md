# Machine Learning for Bioinformatics

Teaching materials for the KSE course. This repository contains the currently released student materials. Future practicals will be added week by week.

## Repository structure

```text
.
├── pyproject.toml
├── uv.lock
├── .python-version
├── lectures/
│   ├── week1/
│   └── week2/
└── practicals/
    ├── week1/
    ├── week2/
    └── week7/
```

Install uv on Windows


```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```


macOS with Homebrew:

```bash
brew install uv
```

macOS without Homebrew:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installing, close and reopen the terminal, then check that `uv` works:

```bash
uv --version
```

## Create the course environment

From the repository root:

```powershell
uv sync --extra cpu
```

This creates or updates the project environment in `.venv` using the versions recorded in `uv.lock`. For a recent NVIDIA GPU, use `uv sync --extra cu128` instead. The `.venv` directory should not be committed to GitHub. See [PYTORCH_SETUP.md](PYTORCH_SETUP.md) for backend selection and verification.

TabPFN is optional because it adds a large tabular foundation-model package and downloads model weights on first use. To enable the optional comparison in the Week 10 notebook, combine it with the selected PyTorch backend:

```powershell
uv sync --extra tabpfn --extra cpu
# or:
uv sync --extra tabpfn --extra cu128
```

The notebook selects the TabPFN-3.5 regression checkpoint and compares dynamic-subset, small-descriptor, and full-descriptor inputs. The first run may require Prior Labs license acceptance and a model-weight download; the default environment remains unchanged if this extra is omitted. For a Windows/Jupyter session, accept the license at `https://ux.priorlabs.ai/account/licenses`, copy the API key from `https://ux.priorlabs.ai/account`, and set it only as a temporary environment variable before starting Jupyter. This avoids the browser callback issue that can produce `WinError 10038`:

```powershell
$env:TABPFN_TOKEN = "paste-your-api-key-here"
uv run jupyter lab
```

Do not write the API key into the notebook or commit it to Git. If an API key was previously pasted into a notebook, revoke/rotate it from the Prior Labs account page.

Start JupyterLab with:

```powershell
uv run jupyter lab
```

The environment includes the libraries used in the current practicals, including scikit-learn, XGBoost, LightGBM, Optuna, TensorFlow, Graphviz, Ultralytics, Cellpose, Napari and the PyTorch CPU/CUDA alternatives.

## CVAT annotation setup

CVAT is used for creating and reviewing object-detection annotations. The recommended setup is Docker Compose: use Docker Desktop with WSL2 on Windows, or Docker Engine and Docker Compose on Linux. CVAT should not be installed as a collection of Python packages inside the course environment.

See [CVAT_SETUP.md](CVAT_SETUP.md) for installation, account creation, task creation and YOLO export instructions.
