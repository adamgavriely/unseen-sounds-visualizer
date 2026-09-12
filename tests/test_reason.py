"""End-to-end stubbed test of reason.decide_subjects: no GPU, no real models.

Every model call is answered by a table, so this checks the CONTROL FLOW -- a visible
source is gated, a synonym is merged, no dialogue reaches a depiction -- on a machine
with no GPU and in under a second. It is not a test of the models; the demo job on the
cluster is what checks those.

    python tests/test_reason.py
""" 
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.stage5_cross_modal_analysis import reason as R
from src.types import AugmentationSpec

SCENE = "A courtroom with a judge and lawyers"
PLACE = "a courtroom"


class M:
    device = "cpu"


def fake_load(model, device):
    return M(), None


def fake_frames(path, n):
    return ["frame"] * n


def fake_frames_at(path, times):
    return ["frame"] * len(times)


ANSWERS = {
    # visibility: gavel is on screen, the laughter is not, the siren is not
    ("visible", "Gavel"): "a wooden gavel",
    ("visible", "Laughter"): "nothing",
    ("visible", "Giggle"): "nothing",
    ("visible", "Siren"): "nothing",
    ("visible", "Speech"): "a lawyer speaking",
    ("visible", "Door"): "nothing",
    ("visible", "Sheep"): "nothing",
    ("visible", "Baby cry, infant cry"): "nothing",
}


def fake_ask(mdl, proc, prompt, images=None, max_new=48):
    if prompt.startswith("Describe this scene"):
        return SCENE
    if prompt.startswith("What kind of place"):
        return PLACE
    if "Judge from the frames alone" in prompt:
        # the gavel is seen striking; answer whichever letter means "action visible"
        lines = prompt.splitlines()
        return "a" if "SEE it making" in [l for l in lines if l.startswith("(a)")][0] else "b"
    if prompt.startswith("These frames are from the moment"):
        label = prompt.split("a sound of ")[1].split(" was heard")[0]
        return ANSWERS.get(("visible", label), "nothing")
    if prompt.startswith("Does a"):
        named = prompt.split("Does a ")[1].split(" make a ")[0]
        label = prompt.split(" make a ")[1].split(" sound")[0]
        return "yes" if (named, label) in {("wooden gavel", "Gavel"),
                                           ("lawyer speaking", "Speech")} else "no"
    if prompt.startswith("A sound detector heard ONE sound"):
        # the frames show a baby: pick whichever letter is Baby cry
        line = [l for l in prompt.splitlines() if l.startswith("Which is it?")][0]
        return "a" if "(a) Baby cry" in line else "b"
    if prompt.startswith("A sound of ") and "someone said" in prompt:
        # speech context: "order, order" is about the gavel AND the door; nothing else
        last = prompt.splitlines()[-1]
        yes = "a" if last.index("reacting") < last.index("not referring") else "b"
        no = "b" if yes == "a" else "a"
        about = ("Gavel" in prompt or "Door" in prompt) and "order" in prompt
        return yes if about else no
    if prompt.startswith("A deaf viewer is watching a video and cannot hear it."):
        label = prompt.split("A sound detector heard: ")[1].split(".")[0].split(" (")[0]
        return {"Laughter": "a group of people laughing in a courtroom",
                "Giggle": "a woman giggling in the courtroom gallery",
                "Siren": "a police car siren outside the courthouse",
                "Door": "a heavy courtroom door slamming shut",
                "Baby cry, infant cry": "a baby crying"}[label]
    if prompt.startswith("A deaf viewer is shown this picture:"):
        # forced choice: pick the option whose label the depiction actually names
        picture = prompt.splitlines()[0].split(": ", 1)[1].lower()
        for line in prompt.splitlines():
            if line.startswith("(") and ") " in line:
                letter, option = line[1], line.split(") ", 1)[1]
                stem = option.lower().split(",")[0][:4]
                if stem and stem in picture:
                    return letter
        return "z"   # none of them
    if prompt.startswith("A sound detector labelled two sounds"):
        last = prompt.splitlines()[-1]
        same = "a" if last.index("the same sound") < last.index("different sounds") else "b"
        diff = "b" if same == "a" else "a"
        return same if ("Laughter" in prompt and "Giggle" in prompt) else diff
    raise AssertionError("unexpected prompt: " + prompt[:70])


R._load = fake_load
R._ask = fake_ask
import src.stage2_video_understanding as S2
S2._sample_frames = fake_frames
S2._sample_frames_at = fake_frames_at


def spec(label, a, b, conf):
    return AugmentationSpec(index=0, event_label=label, start=a, end=b, augment=True,
                            confidence=conf, reason="planned", subject=label)


specs = [spec("Gavel", 1.0, 1.5, 0.7),      # visible -> silent, even though talked about
         spec("Laughter", 3.0, 5.0, 0.6),   # shown
         spec("Giggle", 3.2, 4.0, 0.4),     # merged into Laughter
         spec("Siren", 8.0, 10.0, 0.5)]     # shown
# a faint sound the gate declined; speech about it should rescue it
specs += [spec("Sheep", 12.0, 13.5, 0.30), spec("Baby cry, infant cry", 12.1, 13.6, 0.28)]
faint = spec("Door", 0.8, 1.2, 0.08)
faint.augment = False
faint.reason = "below display threshold (0.08 < 0.12)"
specs.append(faint)

from src.types import SpeechSegment
segs = [SpeechSegment(0, 0.5, 1.8, "order, order in the court"),
        SpeechSegment(1, 6.0, 7.0, "your honour, objection")]
R.decide_subjects("fake.mp4", specs, segments=segs, device="cpu")

print()
print("RESULT")
for s in specs:
    print(("  SHOW   " if s.augment else "  silent ") + s.event_label.ljust(10)
          + "| " + (s.subject or s.reason))
shown = [s.event_label for s in specs if s.augment]
assert shown == ["Laughter", "Siren", "Baby cry, infant cry", "Door"], shown
sheep = next(s for s in specs if s.event_label == "Sheep")
assert not sheep.augment and "frames say" in sheep.reason, "frames should pick the baby over the sheep"
assert faint.talked_about and faint.augment, "speech should rescue the faint door"
gavel = next(s for s in specs if s.event_label == "Gavel")
assert gavel.talked_about and not gavel.augment, "visibility must beat speech"
assert all("objection" not in (s.subject or "") for s in specs)
print("\nOK: visible gated (even when talked about), synonym merged, faint sound rescued by speech, no dialogue in any depiction")
