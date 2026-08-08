"""Tests for HVGOptions resolution (legacy kwargs, merge, conflicts)."""

from __future__ import annotations

import pytest

from scfair.pp._options import HVGOptions, resolve_hvg_options


def test_merged_skips_none_overrides():
    base = HVGOptions(append_budget=50, filter_mito=True)
    out = base.merged(append_budget=None, filter_mito=False)
    assert out.append_budget == 50
    assert out.filter_mito is False


def test_resolve_legacy_only():
    opt = resolve_hvg_options(None, {"append_budget": 100, "filter_ribo": True})
    assert opt.append_budget == 100
    assert opt.filter_ribo is True


def test_resolve_options_only():
    opt = resolve_hvg_options(HVGOptions(append_budget=80), None)
    assert opt.append_budget == 80


def test_resolve_mix_raises():
    with pytest.raises(ValueError, match="Do not mix"):
        resolve_hvg_options(HVGOptions(filter_mito=True), {"append_budget": 50})


def test_resolve_removed_auto_n_method():
    with pytest.raises(TypeError, match="removed option"):
        resolve_hvg_options(None, {"auto_n_method": "ensemble"})


def test_resolve_unknown_raises():
    with pytest.raises(TypeError, match="unknown"):
        resolve_hvg_options(None, {"not_a_real_knob": 1})
