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
