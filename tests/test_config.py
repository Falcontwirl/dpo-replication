import pytest

from dpo_rep.config import load_config


def test_layering_and_overrides(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("dpo:\n  beta: 0.1\n  lr: 1.0e-6\n  name: a\nseed: 0\n")
    over = tmp_path / "over.yaml"
    over.write_text("dpo:\n  beta: 0.5\n")
    cfg = load_config([str(base), str(over)], ["dpo.lr=5e-5", "seed=3", "dpo.name=b"])
    assert cfg.dpo.beta == 0.5
    assert cfg.dpo.lr == 5e-5 and isinstance(cfg.dpo.lr, float)
    assert cfg.seed == 3 and cfg.dpo.name == "b"


def test_unknown_key_rejected(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("dpo:\n  beta: 0.1\n")
    with pytest.raises(KeyError):
        load_config([str(base)], ["dpo.bta=0.5"])
