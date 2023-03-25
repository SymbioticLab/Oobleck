from ast import literal_eval
from typing import Dict, Any
from oobleck.execution.engine import OobleckEngine
from argparse import ArgumentParser


def run():
    parser = ArgumentParser()
    parser.add_argument("--ft_spec", type=int, default=0)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--dataset_name", type=str, required=False, default=None)
    parser.add_argument("--model_args", type=str, required=False, default=None)

    args = parser.parse_args()

    model_args: Dict[str, Any] = (
        literal_eval(args.model_args) if args.model_args is not None else None
    )
    engine = OobleckEngine(
        args.ft_spec, args.model_name, args.dataset_path, args.dataset_name, model_args
    )
    engine.init_distributed()
    engine.train()


if __name__ == "__main__":
    run()
