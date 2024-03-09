# Commands to install a new python version using "pyenv"
# This was needed to deploy the system onto a system that had an old python version installed

# pyenv install
curl https://pyenv.run | bash

# .bashrc setup
echo 'export PYENV_ROOT="$HOME/.pyenv"' >>~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >>~/.bashrc
echo 'eval "$(pyenv init -)"' >>~/.bashrc

# .profile setup
echo 'export PYENV_ROOT="$HOME/.pyenv"' >>~/.profile
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >>~/.profile
echo 'eval "$(pyenv init -)"' >>~/.profile

# Install build deps
sudo apt update
sudo apt install -y build-essential libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev curl \
  libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

# Install python 11
pyenv install 3.11
