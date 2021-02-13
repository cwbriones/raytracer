#!/bin/bash

if [[ ! -e ~/.rustup ]]; then
    echo "Installing rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
else
    echo "Rustup installation found. Updating."
    rustup self update
fi
