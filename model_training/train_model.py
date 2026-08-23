import sys
from omegaconf import OmegaConf
from rnn_trainer import BrainToTextDecoder_Trainer

args = OmegaConf.load(sys.argv[1] if len(sys.argv) > 1 else 'rnn_args.yaml')
trainer = BrainToTextDecoder_Trainer(args)
metrics = trainer.train()