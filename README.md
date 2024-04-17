# Thesis Elias Zielke

## Introduction

The contents of this repository are a product of Elias Zielke's master's thesis: "Computer vision in retail: automated data analysis based on a video surveillance system".

All source and test files that were created as part of the thesis' project are collected here. Furthermore, all project configuration files and python requirements can be found here.

## Structure

The actual source files for the prototype are located inside `./analysis` and `./events`. The remaining files are either test material, configurations files or scripts that handle tasks for deployment, testing or training deep learning models.

To get to know more about each and every part of this repository, corresponding documentations can be used. Every file in this repository is documented right at the beginning (if possible) to make its purpose clear. Every folder is documented with either a `README.md` or an `__init__.py` (for python source code) or both, describing the content of the file/folder.

This project is structered in the "flat layout" style, described by the official packaging documentation: <https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/>.

## Installation

### Requirements

This project uses publicly available third party libraries for Python, called "modules". These modules must be installed in order to provide all needed funccionality, for example from [PyPI](https://pypi.org) using *pip*. How that can be done specifically for this project is described in this part.

Firstly, setup a virtual environment and activate it:

```bash
python -m venv .venv
source .venv/bin/activate
```

Then, install the requirements. For that you need to select a requirements file that matches your installed hardware. The corresponding target hardware will later be used to run deep learning detection or training algorithms via YOLOv8. You can choose from the following:

| Requirements file     | Target Hardware |
| --------------------- | --------------- |
| requirements.cuda.txt | NVIDIA GPU      |
| requirements.rocm.txt | AMD GPU         |
| requirements.cpu.txt  | CPU             |

If you pick the CPU, YOLOv8 will not use a GPU. If you pick, for example, the CUDA requirements but have no NVIDIA GPU installed, the program will default to running on CPU.

Now the actual install can be done (CPU was picked in this example):

```bash
pip install -r requirements.cpu.txt -r requirements.txt
```

The Hardware-specific requirements file has to be used __before__ the other requirements file.

For additional information for this step, see PyTorch installation page: <https://pytorch.org/get-started/locally/>

### Offline Deployment

The following documentation explains how the program be deployed on an offline machine. The only difficulty here is the installation of python requirements.

To make this possible, the requirements can be downloaded to a machine with an internet connection beforehand. These can then be moved to the target machine for installation.

So first, download all dependencies using a requirement selection that matches the target hardware (see [above](#dev-environment)).:

```bash
pip download -r requirements.cpu.txt -r requirements.txt -d requirements/cpu
```

See <https://pip.pypa.io/en/stable/cli/pip_download/> for `pip download` details.

After that, synchronize the requirements onto the target machine.

```bash
rsync -avhrzP --delete requirements/cpu user@hostname:/tmp/requirements
```

This step may take a few minutes (it took 15 for roc in a local test). See <https://www.digitalocean.com/community/tutorialshow-to-use-rsync-to-sync-local-and-remote-directories> for `rsync` details.

Now the actual install can be done like this:

```bash
pip install --no-index --no-build-isolation --find-links /tmp/requirements/ wheel
pip install --no-index --no-build-isolation --find-links /tmp/requirements/ -r requirements.cpu.txt -r requirements.txt
```

See <https://stackoverflow.com/questions/11091623/how-to-install-packages-offline> for comparable forum thread.

## Usage

This project defines two programs, `events` and `analysis`. They both have a detailed CLI, implemented with [Typer](https://typer.tiangolo.com). The CLI can be used to start the programs or read about its features and parameters. This also applies to various Python-Scripts in the `scripts` directory. Python 11 was used for development and testing.

The programs can be run like this:

```bash
# Show help info for the analysis program
PYTHONPATH="." python analysis --help
# Show help info for the events program
PYTHONPATH="." python events --help
# Show help info for the dataset download script
PYTHONPATH="." python scripts/datasets.py --help
# Show help info for the training script
PYTHONPATH="." python scripts/train.py --help
```

The CLIs have submodules and commands, that are all documented with the help info. For example, the following command can be run to get to know more about the "motion-data" submodule:

```bash
PYTHONPATH="." python analysis motion-data --help
```

The next command could be used to get to know more about the "yolo image" command and all its parameters:

```bash
PYTHONPATH="." python analysis yolo image --help
```

Everything should be run from the project root directory. The project root should also be the PYTHONPATH, that is why `PYTHONPATH="."` is set.

This also means that bash scripts should be run like this:

```bash
bash scripts/count.bash
```

The following can be used to run the analysis service with API like it would be run in the service:

```bash
bash ./scripts/run_api.bash
```

## Documentation

As stated in [Structure](#structure), files and folders are documented. In the Python-code, docstrings were used as inline documentation and describe public functions, classes, modules and more. This is also the base for a fully documented HTTP API, using [FastAPI](https://fastapi.tiangolo.com).

After starting the application, for example with `./scripts/run_api_debug.bash` or `./scripts/run_api_only.bash` the following URL can be opened in the browser to see the automatically generated documentation: <http://127.0.0.1:8000/docs>

## Author

As stated in the introduction, this project is part of Elias Zielke's master's thesis. For contact information, please check out <https://elias-zielke.net>.

Thank you for your interest!
