# Thesis Elias Zielke

## Structure

This project is structered in the "flat layout" style, described by the official packaging documentation: <https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/>

## Installation

Firstly, setup a virtual environment and activate it:

```bash
python -m venv .venv
source .venv/bin/activate
```

Then, install `torch` for the intended hardware (this has to be done **before** installing the remaining requirements).

For usage with a NVIDIA GPU:

```bash
pip install -r ./requirements.cuda.txt
```

For usage with an AMD GPU:

```bash
pip install -r ./requirements.rocm.txt
```

For usage with the CPU:

```bash
pip install -r ./requirements.cpu.txt
```

For additional information for this step, see PyTorch installation page: <https://pytorch.org/get-started/locally/>

Now, the remaining requirements can be installed:

```bash
pip install -r ./requirements.txt
```
