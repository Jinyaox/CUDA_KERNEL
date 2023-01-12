import os
import itertools
import contextlib
import csv
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Union,
    Literal,
    NamedTuple,
    List,
    Optional,
)
import os
import torch
import numpy
import random
import itertools
import datetime
from tap import Tap


class Arguments(Tap):
    dataset: List[
        Literal[
            "ogbl-ddi",
            "ogbl-ppa",
            "ogbl-collab",
            "soc-epinions1",
            "soc-livejournal1",
            "soc-pokec",
            "soc-slashdot0811",
            "soc-slashdot0922",
            "facebook",
            "wikipedia",
        ]
    ]
    # the dataset to run the experiment on
    dataset_dir: str = "data"  # directory containing the dataset files
    method: List[
        Literal[
            "jaccard",
            "adamic-adar",
            "common-neighbors",
            "resource-allocation",
            "dothash-jaccard",
            "dothash-adamic-adar",
            "dothash-common-neighbors",
            "dothash-resource-allocation",
            "neuralhash-jaccard",
            "neuralhash-adamic-adar",
            "neuralhash-common-neighbors",
            "neuralhash-resource-allocation",
            "minhash",
            "simhash",
        ]
    ]
    # method to run the experiment with
    dimensions: List[int]
    # number of dimensions to use (does not affect the exact method)
    batch_size: int = 16384  # number of nodes to evaluate at once
    result_dir: str = "results"  # directory to write the results to
    device: List[str] = ["cpu"]  # which device to run the experiment on
    seed: List[int] = [1]  # random number generator seed


class Config(NamedTuple):
    dataset: str  # the dataset to run the experiment on
    method: str  # method to run the experiment with
    dimensions: int  # number of dimensions to use (does not affect the exact method)
    device: torch.device  # which device to run the experiment on
    seed: int  # random number generator seed


class Result(NamedTuple):
    output_pos: torch.Tensor
    output_neg: torch.Tensor
    init_time: float
    calc_time: float
    dimensions: int


METRICS = [
    "method",
    "dataset",
    "dimensions",
    "hits@20",
    "hits@50",
    "hits@100",
    "init_time",
    "calc_time",
    "num_node_pairs",
    "device",
]


@contextlib.contextmanager
def open_metric_writer(filename: str, columns: List[str]):
    """starts a metric writer contexts that will write the specified columns to a csv file"""

    file = open(filename, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(file, columns)

    if os.path.getsize(filename) == 0:
        writer.writeheader()

    def write(metrics: Dict[str, Any]) -> None:
        writer.writerow(metrics)
        file.flush()  # make sure latest metrics are saved to disk

    yield write

    file.close()


def main(conf: Config, args: Arguments, result_file: str):

    torch.manual_seed(conf.seed)
    numpy.random.seed(conf.seed)
    random.seed(conf.seed)

    print("Device:", conf.device)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with tools.open_metric_writer(result_file, METRICS) as write:

        print("Dataset:", conf.dataset)
        dataset = get_dataset(conf.dataset, args.dataset_dir)

        try:
            metrics = get_metrics(conf, args, dataset, device=conf.device)
            metrics["method"] = conf.method
            metrics["dataset"] = conf.dataset
            metrics["device"] = conf.device.type
            write(metrics)
        except Exception as e:
            print(e)


def default_to_cpu(device: str) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(device)
    else:
        return torch.device("cpu")


if __name__ == "__main__":

    args = Arguments(underscores_to_dashes=True).parse_args()

    result_filename = (
        "link_prediction-"
        + datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        + ".csv"
    )

    result_file = os.path.join(args.result_dir, result_filename)
    os.makedirs(args.result_dir, exist_ok=True)

    devices = {default_to_cpu(d) for d in args.device}

    options = (args.seed, devices, args.dimensions, args.dataset, args.method)
    for seed, device, dimensions, dataset, method in itertools.product(*options):
        config = Config(dataset, method, dimensions, device, seed)
        main(config, args, result_file)
