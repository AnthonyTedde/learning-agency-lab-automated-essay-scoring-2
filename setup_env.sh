#!/bin/bash

activate_env() {
    if [[ "$1" == "Linux" ]]; then
        source $2/bin/activate
    else
        source $2/Scripts/activate
    fi
}

OS=$(uname -s)
VENV_PATH=".venv"

activate_env $OS $VENV_PATH

# Install requirements
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
