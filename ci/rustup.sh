#!/bin/bash

if [[ ! -e ~/.rustup ]]; then
    echo "Installing rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > /tmp/install-rustup.sh
    sh /tmp/install-rustup.sh -y
elif [[ ! -e /tmp/rustup-init ]]; then
    echo "rustup installation found. Updating."
    rustup self update
fi

touch /tmp/rustup-init

rustup $@
