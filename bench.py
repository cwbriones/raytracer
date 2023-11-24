#!/usr/bin/env python

import sys
import tempfile
import subprocess
import re
import argparse

def grep(lines, pat):
    for l in lines:
        if (match := pat.search(l)) is not None:
            yield match

def benchmark(n, traceargs=None):
    subprocess.check_call(['cargo', 'build', '--release'])

    if traceargs is None:
        traceargs = ['--width', '300', '--num-samples', '100', '--no-progress']

    benchcmd = ['cargo', 'run', '--release', '--', '-o', 'tmp.png'] + traceargs

    print(f'bench cmd: {" ".join(benchcmd)}')
    pat = re.compile('^Done in ([0-9]+\.[0-9]+)s \(([0-9]+) rays/s\)')
    total_elapsed = 0
    total_rate = 0
    for i in range(n):
        print(f'Run {i+1}/{n}')
        out = subprocess.check_output(benchcmd, stderr=subprocess.STDOUT, text=True)
        m = next(grep(out.split('\n'), pat))
        elapsed = float(m.group(1))
        rate = int(m.group(2))
        print(f'  {elapsed} {rate}')
        total_elapsed += elapsed
        total_rate += rate

    print(f'Average time: {total_elapsed / n:.1f}')
    print(f'Average rate: {total_rate / n:.1f} rays/s')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=15)
    args, traceargs = sys.argv, None
    try:
        sep = sys.argv.index('--')
        traceargs = args[sep+1:]
        args = args[:sep]
    except ValueError:
        pass
    ns = parser.parse_args(args[1:])
    benchmark(ns.n, traceargs)
