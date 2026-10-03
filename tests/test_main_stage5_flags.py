"""main.py must run stage 5 with the scored flags and the picture step with the final flags (no models needed)."""
import config
import main as M
from src import pipeline
from src.stage5_cross_modal_analysis import reason

KEYS = ("KINSHIP_DIRECTED", "PICTURE_V3", "PICTURE_FINAL", "PICTURE_MAKER")


def test_stage5_scored_pictures_final(monkeypatch):
    # config.use_shipped() changes module-level settings: register every setting with monkeypatch so the other
    # tests see the defaults again afterwards
    for k in [k for k in vars(config) if k.isupper()]:
        monkeypatch.setattr(config, k, getattr(config, k))
    seen = {}
    for name, mod in (("plan_augmentations", pipeline), ("generate_augmentations", pipeline)):
        monkeypatch.setattr(mod, name, (lambda n: (lambda *a, **k: seen.__setitem__(n, {x: getattr(config, x) for x in KEYS})))(name))
    monkeypatch.setattr(reason, "decide_subjects", lambda *a, **k: seen.__setitem__("decide_subjects", {x: getattr(config, x) for x in KEYS}))
    config.use_shipped()
    final = {x: getattr(config, x) for x in KEYS}
    M._run_stage5_as_scored()
    pipeline.plan_augmentations()
    reason.decide_subjects()
    after_stage5 = {x: getattr(config, x) for x in KEYS}
    pipeline.generate_augmentations()
    assert all(v is False for v in seen["plan_augmentations"].values())
    assert all(v is False for v in seen["decide_subjects"].values())
    assert seen["generate_augmentations"] == final and after_stage5 == final
    assert final["PICTURE_V3"] is True and final["KINSHIP_DIRECTED"] is True
