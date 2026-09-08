"""YAML configuration support shared by the BOAR and propensity entry points.

Each dataset has one config file, ``configs/<dataset>.yaml``, which is picked up
automatically from ``--dataset``, so a run is just:

    python ./src/main.py --dataset jdata
"""

import argparse
import os

import yaml

# Top-level keys of a config file that hold per-stage settings rather than
# arguments shared by both stages.
SECTIONS = ('propensity', 'boar')


def repo_root():
    """Directory holding configs/, found by walking up from this file."""
    path = os.path.dirname(os.path.abspath(__file__))
    while path != os.path.dirname(path):
        if os.path.isdir(os.path.join(path, 'configs')):
            return path
        path = os.path.dirname(path)
    raise FileNotFoundError('could not locate the configs/ directory')


def config_pre_parser(default_dataset):
    """Parser for the arguments that select a config file, so its values can
    seed the defaults of the full parser."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--dataset', type=str, default=default_dataset, help='Dataset name')
    pre_parser.add_argument('--config', type=str, default=None,
                            help='Config file (default: configs/<dataset>.yaml)')
    return pre_parser


def resolve_config_path(known_args):
    if known_args.config is not None:
        return known_args.config
    path = os.path.join(repo_root(), 'configs', f'{known_args.dataset}.yaml')
    return path if os.path.exists(path) else None


def load_config(config_path, section):
    """Merge the shared keys of a config file with one stage section."""
    with open(config_path, 'r') as f:
        raw = yaml.safe_load(f) or {}

    config = {k: v for k, v in raw.items() if k not in SECTIONS}
    config.update(raw.get(section) or {})
    return config


def apply_config(parser, known_args, section):
    """Use the dataset's config file as argparse defaults (explicit command line
    flags still win)."""
    config_path = resolve_config_path(known_args)
    if config_path is None:
        return

    config = load_config(config_path, section)
    known_keys = {action.dest for action in parser._actions}
    unknown_keys = sorted(set(config) - known_keys)
    if unknown_keys:
        raise ValueError(
            f'Unknown key(s) in {config_path} [{section}]: {", ".join(unknown_keys)}'
        )

    parser.set_defaults(**config)
