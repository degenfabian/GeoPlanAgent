import json
from pathlib import Path

from geoplanagent.utils import normalise_case_name, resolve_fold

ASSIGNMENT = Path("models/sam3_lora/fold_assignment.json")
ALL_FOLDS = {0, 1, 2, 3, 4}


def load_assignment():
    return json.loads(ASSIGNMENT.read_text())


def test_exact_keys_resolve_to_themselves():
    fa = load_assignment()
    for key, fold in fa.items():
        assert resolve_fold(key, fa, ALL_FOLDS) == fold


def test_colon_names_normalise():
    fa = load_assignment()
    assert resolve_fold("12:00114:ART4", fa, ALL_FOLDS) == fa["12_00114_ART4"]
    assert normalise_case_name("12:00114:ART4") == "12_00114_ART4"


def test_multipage_cases_route_to_their_training_fold():
    # Multi-page documents are keyed per page (A108P_p4 etc.) but the
    # benchmark queries with the bare case name. These two used to fall
    # through to fold 0 and get scored by a model that saw their GT.
    fa = load_assignment()
    assert resolve_fold("A108P", fa, ALL_FOLDS) == fa["A108P_p4"]
    assert resolve_fold("A4D6A_merged", fa, ALL_FOLDS) == fa["A4D6A_merged_p9"]


def test_unseen_case_falls_back_deterministically():
    fa = load_assignment()
    assert resolve_fold("not_in_training_pool", fa, ALL_FOLDS) == 0
    assert resolve_fold("not_in_training_pool", fa, {2, 4}) == 2
