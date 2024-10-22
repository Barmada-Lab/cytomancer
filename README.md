## What's this?

cytomancer is a collection of tools designed around the needs of Sami Barmada's Lab at the University of Michigan. The contents of this collection are mostly oriented towards bioimage analysis.

## Requirements

  - python 3.11

## Installation

### Using pipx (recommended)

The simplest way to use these tools is to install them with pipx. If you don't have pipx installed already, you can install it by running:

    python -m pip install --user pipx

Then, install the repository by running:

    pipx install git+https://github.com/Barmada-Lab/cytomancer

The main entrypoint to cytomancer should then be made available via the `cyto` command.

## Usage

You can launch a Terminal User Interface (TUI) session by running:

    cyto tui

Alternatively, you can invoke any command listed in the tui directly from your shell. For information about the commands available, run:

    cyto --help
