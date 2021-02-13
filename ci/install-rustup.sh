#!/bin/bash

if [[ ! -e ~/.rustup ]]; then
    echo "rustup installation not found, running rustup-init..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > /tmp/install-rustup.sh
    sh /tmp/install-rustup.sh -y
elif [[ ! -e /tmp/rustup-updated ]]; then
    echo "rustup installation found. updating."
    rustup self update
fi

touch /tmp/rustup-updated
