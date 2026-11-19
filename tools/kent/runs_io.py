"""Read Kent's run folders (results_search/<rc>/carla/<batch>/[repeats/]runs/<run_id>/).

In the repository the per-tick files are gzipped: ground_truth.csv.gz, ego_observable.csv.gz, decisions.jsonl.gz;
summary.json is plain. Numbers with more than 7 significant digits were written at float32 precision (CARLA's own;
relative change < 5e-7). These readers take a run folder and open either the .gz or the plain file.

    from tools.kent.runs_io import summary, ground_truth, ego_observable, decisions
    s = summary(run_dir); gt = ground_truth(run_dir)          # dict; pandas DataFrame
"""
import gzip
import json
import os


def _path(run_dir, name):
    for p in (os.path.join(run_dir, name + ".gz"), os.path.join(run_dir, name)):
        if os.path.exists(p):
            return p
    raise FileNotFoundError(os.path.join(run_dir, name) + "[.gz]")


def open_text(run_dir, name):
    """A text handle on <name> or <name>.gz in run_dir."""
    p = _path(run_dir, name)
    return gzip.open(p, "rt") if p.endswith(".gz") else open(p)


def summary(run_dir):
    with open_text(run_dir, "summary.json") as fh:
        return json.load(fh)


def decisions(run_dir):
    with open_text(run_dir, "decisions.jsonl") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _csv(run_dir, name):
    import pandas as pd
    return pd.read_csv(_path(run_dir, name))            # pandas infers the gzip compression from the name


def ground_truth(run_dir):
    return _csv(run_dir, "ground_truth.csv")


def ego_observable(run_dir):
    return _csv(run_dir, "ego_observable.csv")
