"""Command-line interface matching the subject's examples.

    python mybci.py <subject> <run> train
    python mybci.py <subject> <run> predict
    python mybci.py                      # evaluate all subjects / experiments
"""

import argparse


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="mybci")
    parser.add_argument("subject", type=int, nargs="?")
    parser.add_argument("run", type=int, nargs="?")
    parser.add_argument("mode", choices=["train", "predict"], nargs="?")
    args = parser.parse_args(argv)

    given = [args.subject, args.run, args.mode]
    if any(v is not None for v in given) and None in given:
        parser.error("provide either no arguments or <subject> <run> <mode>")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.mode is None:
        raise NotImplementedError("full evaluation not implemented yet")
    raise NotImplementedError(f"{args.mode} not implemented yet")
