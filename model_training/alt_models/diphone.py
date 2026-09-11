"""Diphone (context-dependent phoneme) targets for CTC.

Motivation (DCoND, arXiv:2411.10657).  A monophone is acoustically ambiguous:
its realisation depends heavily on what comes before and after it.  If the
network predicts the phoneme *transition* instead of the isolated phoneme, the
CTC targets become more acoustically coherent, and the same GRU reaches a lower
PER for free -- no architecture change at all.

Why this is a drop-in change
----------------------------
A diphone posterior contains the identity of both of its constituents, so the
monophone posterior can be recovered exactly by marginalisation:

    P(p_k) = 0.5 * ( sum_j P(p_k, p_j)  +  sum_i P(p_i, p_k) )

After that marginalisation the model emits a plain `n_classes`-way phoneme
posterior, which is byte-for-byte what `rnn_model.GRUDecoder` emits.  Everything
downstream -- the greedy CTC decode, PER, the saved validation logits, the
`rearrange_speech_logits_pt` reordering, the WFST language model -- is therefore
completely untouched.

Class layout (must match the baseline contract seen by the rest of the repo)
---------------------------------------------------------------------------
    class 0                       CTC blank            (unchanged)
    class 1 .. n_symbols          the real symbols     (phonemes + <sil>)
    class 1 + s*S .. 1 + s*S + S  diphone classes, id = 1 + left * S + right

where `S = n_symbols = n_phoneme_classes - 1` and `left`/`right` are 0-based
symbol indices (`symbol_index = class_id - 1`).

For the 35-class English set (`<blank>` + 33 phonemes + `<sil>`) this gives
`S = 34` and `1 + 34*34 = 1157` output classes.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch


def n_symbols_for(n_phoneme_classes: int) -> int:
    """Number of non-blank symbols in a baseline `n_classes`-way phoneme head."""
    if n_phoneme_classes < 2:
        raise ValueError(
            f"n_phoneme_classes must be at least 2 (blank + one symbol), got {n_phoneme_classes}"
        )
    return int(n_phoneme_classes) - 1


def n_diphone_classes(n_phoneme_classes: int) -> int:
    """Total output width of a diphone head: blank + n_symbols**2."""
    s = n_symbols_for(n_phoneme_classes)
    return 1 + s * s


def diphone_id(left_symbol: int, right_symbol: int, n_symbols: int) -> int:
    """Class id of the diphone `(left_symbol, right_symbol)`, both 0-based."""
    if not (0 <= left_symbol < n_symbols):
        raise ValueError(f"left_symbol {left_symbol} out of range [0, {n_symbols})")
    if not (0 <= right_symbol < n_symbols):
        raise ValueError(f"right_symbol {right_symbol} out of range [0, {n_symbols})")
    return 1 + left_symbol * n_symbols + right_symbol


def encode_phoneme_sequence(
    phoneme_ids: Sequence[int],
    n_symbols: int,
    close_with_sil: bool = False,
    sil_phoneme_id: Optional[int] = None,
) -> list:
    """Convert one phoneme class-id sequence into diphone class ids.

    `phoneme_ids` are baseline class ids (1..n_symbols).  With
    `close_with_sil=False` a length-L sequence yields L-1 diphones, which is the
    textbook DCoND formulation and is strictly easier for CTC than the baseline
    (which needed L frames).  With `close_with_sil=True` the sequence is closed
    with `<sil>`, yielding L diphones and exactly the baseline's target length.

    This Python helper exists for tests and for one-off inspection.  The
    training loop uses the vectorised `batch_phonemes_to_diphones` instead.
    """
    ids = [int(x) for x in phoneme_ids]
    if not ids:
        return []
    if close_with_sil:
        sil_id = n_symbols if sil_phoneme_id is None else int(sil_phoneme_id)
        ids = ids + [sil_id]
    return [
        diphone_id(ids[i] - 1, ids[i + 1] - 1, n_symbols)
        for i in range(len(ids) - 1)
    ]


def batch_phonemes_to_diphones(
    labels: torch.Tensor,
    lengths: torch.Tensor,
    n_symbols: int,
    close_with_sil: bool = False,
    sil_phoneme_id: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorised phoneme-batch -> diphone-batch conversion.

    Parameters
    ----------
    labels  : (B, Lmax) integer class ids, right-padded with the CTC blank (0)
    lengths : (B,) true phoneme length of each row
    n_symbols : number of non-blank symbols (n_classes - 1)

    Returns
    -------
    diphone_labels : (B, Dmax) integer diphone class ids, right-padded with 0
    diphone_lengths: (B,) true diphone length of each row

    Padded entries are 0 (the CTC blank) and are masked out by
    `diphone_lengths`, which is exactly the convention `nn.CTCLoss` expects.
    """
    if labels.dim() != 2:
        raise ValueError(f"labels must be 2-D (B, Lmax), got shape {tuple(labels.shape)}")
    if lengths.dim() != 1 or lengths.shape[0] != labels.shape[0]:
        raise ValueError(
            f"lengths must be 1-D with one entry per row; got {tuple(lengths.shape)} "
            f"for labels of shape {tuple(labels.shape)}"
        )

    b, lmax = labels.shape
    lengths = lengths.to(device=labels.device, dtype=torch.long)

    if close_with_sil:
        sil_id = n_symbols if sil_phoneme_id is None else int(sil_phoneme_id)
        left = labels
        right = torch.zeros_like(labels)
        if lmax > 1:
            right[:, :-1] = labels[:, 1:]
        # The diphone at position i is (labels[i], labels[i+1]); the last real
        # position L-1 has no successor, so its right-hand slot becomes <sil>.
        # That is index L-1, not L -- the clamp only guards the degenerate
        # L == 0 case, which cannot reach here for real data.
        right.scatter_(
            1,
            (lengths - 1).clamp(min=0, max=lmax - 1).unsqueeze(1),
            torch.full((b, 1), sil_id, dtype=labels.dtype, device=labels.device),
        )
        positions = lengths
        dmax = int(lmax)
    else:
        left = labels[:, :-1]
        right = labels[:, 1:]
        positions = (lengths - 1).clamp(min=0)
        dmax = max(1, lmax - 1)

    if lmax < 2:
        # Degenerate 1-column input; nothing to pair up.
        diphone_labels = torch.zeros((b, 1), dtype=labels.dtype, device=labels.device)
        return diphone_labels, torch.zeros_like(lengths)

    # Only positions whose right-hand neighbour is inside the real sequence are
    # valid. Padded positions carry a 0 on either side, which would otherwise
    # produce a bogus diphone id, so mask them back to the blank afterwards.
    arange = torch.arange(dmax, device=labels.device).unsqueeze(0)
    valid = arange < positions.unsqueeze(1)

    left = left[:, :dmax].to(torch.long)
    right = right[:, :dmax].to(torch.long)
    ids = 1 + (left - 1) * n_symbols + (right - 1)
    ids = torch.where(valid, ids, torch.zeros_like(ids))

    return ids, positions


def marginalize_diphone_logits(
    diphone_logits: torch.Tensor,
    n_phoneme_classes: int,
) -> torch.Tensor:
    """Marginalise `(..., 1 + S*S)` diphone logits to `(..., n_classes)` phoneme logits.

    Computed in log space with `logsumexp` / `logaddexp`, so it is numerically
    stable and fully differentiable:

        out[..., 0]     = D[..., 0]                                   (blank)
        out[..., k + 1] = logaddexp( logsumexp_j D[..., k, j],
                                     logsumexp_i D[..., i, k] ) - log 2

    `softmax(out)` is then exactly `0.5 * (sum_j P[k,j] + sum_i P[i,k])`, the
    standard diphone marginalisation, and it sums to 1 across all n_classes
    entries.  Working in log space (rather than materialising a
    `(..., n_classes, n_diphone)` matrix) keeps the memory cost flat -- the
    naive version would need ~5 GB for a 64x500 batch.
    """
    if diphone_logits.shape[-1] != n_diphone_classes(n_phoneme_classes):
        raise ValueError(
            f"expected {n_diphone_classes(n_phoneme_classes)} diphone logits "
            f"(1 + {n_symbols_for(n_phoneme_classes)}**2), got {diphone_logits.shape[-1]}"
        )

    s = n_symbols_for(n_phoneme_classes)
    blank = diphone_logits[..., :1]
    block = diphone_logits[..., 1:].reshape(*diphone_logits.shape[:-1], s, s)

    left_role = torch.logsumexp(block, dim=-1)   # sum over the right-hand symbol
    right_role = torch.logsumexp(block, dim=-2)  # sum over the left-hand symbol
    merged = torch.logaddexp(left_role, right_role) - math.log(2.0)

    return torch.cat([blank, merged], dim=-1)


def phoneme_logits_to_diphone_argmax(
    phoneme_ids: torch.Tensor,
    n_symbols: int,
) -> torch.Tensor:
    """Greedy self-diphone encoding, used only by the unit tests."""
    return torch.tensor(
        [diphone_id(int(p) - 1, int(p) - 1, n_symbols) for p in phoneme_ids.tolist()],
        dtype=torch.long,
    )
