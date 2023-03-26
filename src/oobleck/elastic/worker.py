import os
import redis

from ast import literal_eval
from typing import Dict, Tuple, List, Optional, Any, TypeVar
from oobleck.execution.engine import OobleckEngine
from argparse import ArgumentParser, Namespace

T = TypeVar("T", bound="ElasticWorker")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--ft_spec", type=int, default=0)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--dataset_name", type=str, required=False, default=None)
    parser.add_argument("--model_args", type=str, required=False, default=None)

    return parser.parse_args()


class ElasticWorker:
    def __init__(self):
        super().__init__()

    def run(
        self,
        ft_spec: int,
        model_name: str,
        dataset_path: str,
        dataset_name: Optional[str] = None,
        model_args: Optional[Dict[str, Any]] = None,
    ):
        self.engine = OobleckEngine(
            ft_spec, model_name, dataset_path, dataset_name, model_args
        )
        self.engine.subscribe_reconfiguration()
        self.engine.train()


if __name__ == "__main__":
    worker = ElasticWorker()

    args = parse_args()
    model_args: Dict[str, Any] = (
        literal_eval(args.model_args) if args.model_args is not None else None
    )

    worker.run(
        args.ft_spec, args.model_name, args.dataset_path, args.dataset_name, model_args
    )
