"""K-fold case-name routing shared by SAM3 and the rotation classifier.

Both production models (SAM3 LoRA + ResNet50 rotation classifier) are
trained with a 5-fold cross-validation split. At inference time the
benchmark routes each case to the checkpoint whose held-out fold
contains that case — guaranteeing the model never saw the case during
training. This module is the single source of truth for that routing
logic.

Previously the same three helpers lived in both
``tools.extraction.sam3`` and ``tools.io.rotation_classifier`` (with
explicit comments noting the duplication was intentional to keep
dependencies light). The dependency-weight argument no longer holds —
both files already pull in torch — so the duplication is now just
duplication.
"""

from __future__ import annotations

import hashlib

# Default fold count. Both SAM3 and the rotation classifier were
# trained with 5 folds; keep this in sync with the build scripts in
# scripts/build_*_training_set.py.
N_FOLDS = 5


def normalise_case_name(case_name: str) -> str:
    """Canonical underscore form of a case identifier. Idempotent.

    The auto-labeller and curated-dataset builder use a 'safe filename'
    convention that replaces ``:`` and ``/`` with ``_``. The benchmark
    runner passes the original eval-data folder name (which may contain
    colons), so we have to translate before any lookup or hash —
    without this, lookups miss and the fallback hash on the colon form
    differs from the hash on the underscore form, producing silent
    leakage.
    """
    return (case_name or "").replace(":", "_").replace("/", "_")


def fold_for_case(case_name: str, n_folds: int = N_FOLDS) -> int:
    """Deterministic fold assignment via md5(canonical_case_name) % n_folds.

    Mirrors ``scripts/build_curated_training_set.py:fold_for`` so a case
    that was in fold k's val set during training also routes to fold k
    at inference (= the model that did NOT see this case during training).

    IMPORTANT: hash the canonical (underscore) form so that
    ``md5('12:00114:ART4')`` and ``md5('12_00114_ART4')`` resolve to the
    same fold — both are aliases for the same case.
    """
    canonical = normalise_case_name(case_name)
    h = hashlib.md5(canonical.encode()).hexdigest()
    return int(h, 16) % n_folds


def resolve_fold(case_name: str, fold_assignment: dict,
                  available_folds) -> int:
    """Pick the fold index to use for ``case_name``.

    Three-step resolution:
      1. Look up the case in the trained-time assignment file
         (handles cases that were in fold k's val set).
      2. If missing, retry with the canonical underscore form
         (legacy assignment files use the safe-filename key).
      3. If still missing, fall back to the deterministic md5 hash —
         keeps inference working for cases that weren't part of the
         original train set.

    Finally, clamp to ``available_folds`` so a missing/un-trained fold
    (e.g. only folds {0, 1, 2} on disk) doesn't blow up — pick the
    lowest available instead.

    The explicit ``is None`` checks (not ``or``) matter: fold 0 is a
    valid assignment that an ``or`` chain would treat as falsy and
    silently re-route via the md5 fallback.
    """
    fold = fold_assignment.get(case_name)
    if fold is None:
        fold = fold_assignment.get(normalise_case_name(case_name))
    if fold is None:
        fold = fold_for_case(case_name)
    if fold not in available_folds:
        fold = min(available_folds)
    return int(fold)
