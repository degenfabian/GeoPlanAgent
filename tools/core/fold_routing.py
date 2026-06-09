"""K-fold case routing shared by SAM3 and the rotation classifier.

Both models are trained 5-fold; at inference each case must be routed to
the checkpoint that held it out, so no case is ever scored by a model
that saw its ground truth.
"""

from __future__ import annotations

N_FOLDS = 5


def normalise_case_name(case_name: str) -> str:
    """Map a case name to the safe-filename form used in fold_assignment.json.

    The dataset builder replaces ':' and '/' with '_', so e.g. the eval
    folder '12:00114:ART4' is keyed as '12_00114_ART4'.
    """
    return (case_name or "").replace(":", "_").replace("/", "_")


def resolve_fold(case_name: str, fold_assignment: dict,
                 available_folds: set[int]) -> int:
    """Return the fold whose checkpoint should serve `case_name`.

    Lookup order: exact key, then the normalised safe-filename form, then
    page-suffixed keys (multi-page cases are stored per page, e.g.
    'A108P_p4', but the benchmark asks for 'A108P'). Cases outside the
    training pool were never seen by any fold, so any checkpoint is fine;
    we pick min(available_folds) for determinism.
    """
    norm = normalise_case_name(case_name)
    fold = fold_assignment.get(case_name)
    if fold is None:
        fold = fold_assignment.get(norm)
    if fold is None:
        # Multi-page cases: pages of one document always share a fold
        # (the split is grouped by case), so the first hit is enough.
        prefix = norm + "_p"
        for key, val in fold_assignment.items():
            if key.startswith(prefix) and key[len(prefix):].isdigit():
                fold = val
                break
    if fold is None or fold not in available_folds:
        return min(available_folds)
    return int(fold)
