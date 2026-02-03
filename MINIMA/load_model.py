# Modified from https://github.com/LSXI7/MINIMA (Apache License 2.0).
# Changes vs upstream:
#   - Trimmed to LoFTR backbone only (RoMa / SP+LG / XoFTR loaders removed).
#   - Added MPS device detection (`_best_device`) for Apple Silicon.
import logging
import os
import torch
from copy import deepcopy


def _best_device():
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_loftr(args, test_orginal_megadepth=False):
    from third_party.LoFTR_minima.src.loftr import LoFTR, default_cfg
    if test_orginal_megadepth:
        from src.config.default_for_megadepth_dense import get_cfg_defaults
    else:
        from src.config.default import get_cfg_defaults
    from src.utils.data_io_loftr import DataIOWrapper, lower_config
    config = get_cfg_defaults(inference=True)
    config = lower_config(config)
    _default_cfg = deepcopy(default_cfg)
    filename = os.path.basename(args.ckpt)
    if filename != "outdoor_ds.ckpt":
        # not using official old model; flip the temp_bug_fix
        _default_cfg['coarse']['temp_bug_fix'] = True

    _default_cfg['match_coarse']['thr'] = args.thr
    matcher = LoFTR(config=_default_cfg)
    _map_loc = _best_device()
    matcher.load_state_dict(
        torch.load(args.ckpt, map_location=_map_loc)['state_dict'], strict=True,
    )
    matcher = matcher.eval()

    matcher = DataIOWrapper(matcher, config=config["test"])
    logging.info(config["test"])
    return matcher


def load_model(method, args, use_path=True, test_orginal_megadepth=False):
    if method == "loftr":
        matcher = load_loftr(args, test_orginal_megadepth=test_orginal_megadepth)
    else:
        raise ValueError(
            f"load_model: only 'loftr' is supported. "
            f"Requested method={method!r}."
        )
    return matcher.from_paths if use_path else matcher.from_cv_imgs


def choose_method_arguments(parser):
    parser.add_argument(
        '--method', type=str, default='loftr', choices=['loftr'],
        help="loftr",
    )


def add_method_arguments(parser, method):
    if method == "loftr":
        parser.add_argument('--ckpt', type=str,
                            default="./weights/minima_loftr.ckpt")
        parser.add_argument('--thr', type=float, default=0.2)
    else:
        raise ValueError(f"Unknown method: {method}")
