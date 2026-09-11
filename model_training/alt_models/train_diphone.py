"""Train a diphone-target GRU.  Drop-in sibling of `model_training/train_model.py`.

    cd Speech/model_training
    python alt_models/train_diphone.py alt_models/rnn_args_diphone.yaml

The `if __name__ == "__main__"` guard matters on Windows: `DataLoader` with
`num_dataloader_workers > 0` uses the *spawn* start method there, which
re-executes the entry script in every worker.  Unguarded, each worker would
re-run `DiphoneTrainer(...)` and die on `os.makedirs(output_dir,
exist_ok=False)`.  Under WSL/Linux the default is *fork* and it makes no
difference, but the guard costs nothing and keeps the script portable.
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
for _p in (str(_PARENT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main() -> None:
    from omegaconf import OmegaConf

    from diphone_trainer import DiphoneTrainer

    args = OmegaConf.load(
        sys.argv[1] if len(sys.argv) > 1 else str(_HERE / "rnn_args_diphone.yaml")
    )
    trainer = DiphoneTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
