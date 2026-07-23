"""One-agent development smoke run, executable from a source checkout."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.pretrain_llm import train_one_step


if __name__ == "__main__":
    print(train_one_step())
