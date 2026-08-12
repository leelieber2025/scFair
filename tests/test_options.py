"""Tests for HVGOptions resolution."""

from __future__ import annotations

import pytest

from scfair.pp._options import HVGOptions, resolve_hvg_options


def test_merged_skips_none_overrides():
    base = HVGOptions(append_budget=50, filter_mito=True)
    out = base.merged(append_budget=None, filter_mito=False)
    assert out.append_budget == 50
    assert out.filter_mito is False


def test_resolve_none_is_defaults():
    opt = resolve_hvg_options(None)
    assert isinstance(opt, HVGOptions)
    assert opt.append_budget is None


def test_resolve_options_only():
    opt = resolve_hvg_options(HVGOptions(append_budget=80))
    assert opt.append_budget == 80


def test_resolve_dict_raises():
    with pytest.raises(TypeError, match="HVGOptions"):
        resolve_hvg_options({"append_budget": 50})
