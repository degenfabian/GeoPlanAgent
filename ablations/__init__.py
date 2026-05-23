"""Ablation harnesses for the planning-boundary pipeline.

Scripts in this directory are run from the repo root, e.g.

    uv run python ablations/locate_only_eval.py

Each script adds the repo root to sys.path so ``tools.*`` imports work
when invoked directly. The ``__init__.py`` exists so they can also
import each other / shared helpers under ``ablations._shared``.
"""
