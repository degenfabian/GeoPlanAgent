"""K-fold case-name routing shared by SAM3 and the rotation classifier.

Both production models (SAM3 LoRA + ResNet50 rotation classifier) are
trained with a 5-fold cross-validation split. At inference time the
benchmark routes each case to the checkpoint whose held-out fold
contains that case — guaranteeing the model never saw the case during
training. This module is the single source of truth for that routing
logic.
"""

from __future__ import annotations


# Default fold count. Both SAM3 and the rotation classifier were
# trained with 5 folds; keep this in sync with the build scripts.
N_FOLDS = 5


def normalise_case_name(case_name: str) -> str:
    """Canonical underscore form of a case identifier. Idempotent.

    The auto-labeller and curated-dataset builder use a 'safe filename'
    convention that replaces ``:`` and ``/`` with ``_``. The benchmark
    runner passes the original eval-data folder name (which may contain
    colons), so we have to translate before any lookup — without this,
    lookups miss for cases like ``12:00114:ART4`` whose fold_assignment
    key is ``12_00114_ART4``.
    """
    return (case_name or "").replace(":", "_").replace("/", "_")


def resolve_fold(case_name: str, fold_assignment: dict,
                  available_folds: set[int]) -> int:
    """Pick the fold index to use for ``case_name``.

    ``available_folds`` is a set of int fold indices that actually have
    a usable adapter on disk. We rely on O(1) membership tests, so
    callers must pass a set (not a list).

    Resolution order:
      1. Look up the case in the trained-time assignment file (handles
         cases that were in fold k's val set during training).
      2. If missing, retry with the canonical underscore form (covers
         the colon→underscore safe-filename rewrite).
      3. If still missing, return ``min(available_folds)`` — the case
         wasn't in our training pool, so no fold "owns" it; any fold's
         adapter is equally valid (none of them saw it). Pick
         deterministically rather than via hash; the hash carried no
         real signal since unseen cases have no preferred fold.

    Finally, clamp to ``available_folds`` so a missing/un-trained fold
    (e.g. only folds {0, 1, 2} on disk) doesn't blow up.

    The explicit ``is None`` checks (not ``or``) matter: fold 0 is a
    valid assignment that an ``or`` chain would treat as falsy.
    """
    fold = fold_assignment.get(case_name)
    if fold is None:
        fold = fold_assignment.get(normalise_case_name(case_name))
    if fold is None or fold not in available_folds:
        return min(available_folds)
    return int(fold)
