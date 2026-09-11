"""Alternative decoders for the NEJM brain-to-text pipeline.

Everything in this package is additive: no file under `model_training/` is
modified, and the original entry points (`train_model.py`, `evaluate_model.py`,
their `rnn_args_*.yaml` configs and any existing checkpoint) keep working
exactly as before.

Currently provided
------------------
`diphone.py` / `diphone_model.py` / `diphone_trainer.py`
    Diphone (context-dependent phoneme) CTC targets, after DCoND
    (arXiv:2411.10657).  The head predicts phoneme transitions; the monophone
    posterior is recovered by exact marginalisation, so the model is
    output-compatible with `GRUDecoder` and the language-model evaluation path
    is untouched.
"""
