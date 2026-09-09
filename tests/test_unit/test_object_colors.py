"""Colour picking for label/suffix objects (object_color_mode)."""

import numpy as np
import pytest

from octron import config
from octron.sam_octron.object_organizer import Obj, ObjectOrganizer


@pytest.fixture
def cfg_path(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    monkeypatch.setenv("OCTRON_CONFIG_PATH", str(path))
    return path


def _colors(mode):
    config.set_value("object_color_mode", mode)
    org = ObjectOrganizer()
    org.add_entry(0, Obj(label="bird", suffix="male"))
    org.add_entry(1, Obj(label="bird", suffix="female"))
    org.add_entry(2, Obj(label="mouse", suffix="1"))
    return {e.suffix: np.array(e.color[:3]) for e in org.entries.values()}


def test_label_mode_keeps_suffixes_in_one_family(cfg_path):
    c = _colors("label")
    same_label = np.abs(c["male"] - c["female"]).max()
    other_label = np.abs(c["male"] - c["1"]).max()
    assert same_label < other_label


def test_individual_mode_separates_suffixes(cfg_path):
    c = _colors("individual")
    same_label = np.abs(c["male"] - c["female"]).max()
    assert same_label > 0.3
    assert np.abs(c["female"] - c["1"]).max() > 0.3


def test_explicit_color_is_kept(cfg_path):
    config.set_value("object_color_mode", "individual")
    org = ObjectOrganizer()
    org.add_entry(0, Obj(label="bird", suffix="a", color=[1, 0, 0, 1]))
    assert org.entries[0].color == [1, 0, 0, 1]


def test_invalid_mode_rejected(cfg_path):
    with pytest.raises(ValueError):
        config.set_value("object_color_mode", "rainbow")
