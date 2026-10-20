"""Compatibility entry point for the local-only research workflow."""
import argparse

from src.validation.local_workflow import run as local_workflow


def run(download=False):
    return local_workflow(refresh=download)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Refresh public SEC sources locally; never deploy AWS")
    run(**vars(parser.parse_args()))
