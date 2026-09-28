"""Ontology-sense traps (Adam, 28 Sept 2026): a label word is read in its AudioSet ontology sense, never its plain-English
one. "Honk" is a goose's call (Honk > Goose > Fowl), not a car horn. No GPU, no model: the maker rule is run with no
frames (the "unsure" path) and the checker's table lookup is pure.

    python tests/test_ontology_traps.py        (or pytest)
"""
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.labels import ontology_maker, ontology_senses, sense_consistent, maker_name, ACTION_LABELS, NON_MAKERS, _parents
from src.stage5_cross_modal_analysis import reason as R
from src.stage6_visual_augmentation import verify as V

# ONE sense: label -> (the only thing that makes it, the wrong plain-English reading it must never take)
TRAPS = {
    "Honk": ("Goose", "a car horn"),
    "Toot": ("Car", "a flute or a whistle"),
    "Bark": ("Dog", "tree bark"),
    "Bay": ("Dog", "a bay of the sea, a bay window"),
    "Caw": ("Crow", "-"),
    "Crowing, cock-a-doodle-doo": ("Chicken, rooster", "a crow (the bird)"),
    "Patter": ("Rodents, rats, mice", "rain"),
    "Roar": ("Roaring cats (lions, tigers)", "an engine, a crowd, the sea"),
    "Chop": ("Wood", "chopping food"),
    "Coo": ("Pigeon, dove", "-"),
    "Purr": ("Cat", "an engine"),
    "Nicker": ("Horse", "underwear"),
    "Gobble": ("Turkey", "eating fast"),
    "Squawk": ("Bird", "a radio"),
    "Air horn, truck horn": ("Truck", "a hand-held air horn"),
    "Vehicle horn, car horn, honking": ("Car", "a trumpet-shaped horn"),
    "Train horn": ("Train", "a horn on its own"),
}
# SEVERAL senses (all parents of the official ontology, src/audioset_ontology.json): no forced maker; None = a
# source-less sense (Onomatopoeia, Brief tone, Generic impact sounds, Clicking)
MULTI = {
    "Hiss": ["Cat", "Snake", "Steam", None],
    "Rattle": ["Snake", None],
    "Growling": ["Dog", "Cat", "Roaring cats (lions, tigers)", "Canidae, dogs, wolves"],
    "Buzz": ["Fly, housefly", "Bee, wasp, etc.", None],
    "Tap": ["Door", None],
    "Knock": ["Door", None],
    "Squeak": ["Door", None],
    "Tick": ["Clock", None],
    "Crack": ["Wood", None],
    "Snap": ["Wood", None],
    "Clip-clop": ["Horse", None],
    "Bleat": ["Goat", "Sheep"],
    "Howl": ["Dog", "Canidae, dogs, wolves"],
    "Crackle": ["Fire", None],
    "Chirp, tweet": ["Bird", None],
}
# no thing in any sense: never given an invented maker (a burst card or the subject as it is)
NO_MAKER = ["Bang", "Thump, thud", "Thunk", "Whoosh, swoosh, swish", "Clunk", "Wobble", "Ding", "Burst, pop",
            "Trickle, dribble", "Gush", "Squish"]


def spec(label, source=None):
    return NS(event_label=label, source=source or label, index=0, start=0.0, subject="")


def test_maker_is_the_ontology_sense():
    for lab, (maker, _) in TRAPS.items():
        assert ontology_maker(lab) == maker, (lab, ontology_maker(lab), maker)


def test_several_senses_from_all_parents():
    for lab, senses in MULTI.items():
        assert ontology_senses(lab) == senses, (lab, ontology_senses(lab))
        assert ontology_maker(lab) is None, lab


def test_several_senses_never_guess():
    # no frames, nothing co-detected: the subject stays as it is
    assert R.with_maker(spec("Hiss"), "hiss", None, None, None) == "hiss"
    assert R.with_maker(spec("Rattle"), "rattle", None, None, None) == "rattle"
    assert R.with_maker(spec("Dog", "Growling"), "growling", None, None, None) == "growling"
    # a co-detected parent decides: steam heard with the hiss, a cat heard with it
    assert R.with_maker(spec("Hiss"), "hiss", None, None, None, codetected=["Steam"]) == "a jet of steam making a hiss sound"
    assert R.with_maker(spec("Hiss"), "A cat hissing", None, None, None, codetected=["Cat"]) == "A cat hissing"
    assert R.with_maker(spec("Hiss"), "A snake hissing", None, None, None, codetected=["Meow"]) ==         "a cat making a hiss sound"                                   # Meow is a kind of Cat
    # the frames decide (a stub model answers the closed question)
    old = R._ask
    try:
        R._ask = lambda *a, **k: "snake"
        assert R.with_maker(spec("Rattle"), "rattle", ["f"], object(), None) == "a snake making a rattle sound"
        R._ask = lambda *a, **k: "unsure"
        assert R.with_maker(spec("Rattle"), "rattle", ["f"], object(), None) == "rattle"
    finally:
        R._ask = old


def test_no_invented_maker():
    for lab in NO_MAKER:
        assert ontology_maker(lab) is None, (lab, ontology_maker(lab))


def test_lists_name_real_labels():
    par = _parents()
    known = set(par) | set(par.values())
    missing = sorted(x for x in ACTION_LABELS | NON_MAKERS if x not in known)
    # Footsteps is a pipeline family name; Channel... is the environment branch
    assert set(missing) <= {"Footsteps", "Channel, environment and background"}, missing


def test_honk_is_a_goose_unless_frames_show_a_car():
    s = spec("Honk")
    assert not sense_consistent("Car", "Honk") and sense_consistent("Goose", "Honk")
    # the text-only subject named a car; no frames -> the ontology sense wins
    assert R.with_maker(s, "A car honking", None, None, None) == "a goose honking with its beak open"
    assert R.with_maker(s, "honk", None, None, None) == "a goose honking with its beak open"
    assert R.with_maker(s, "A goose honking", None, None, None) == "A goose honking"


def test_horn_keeps_its_vehicle():
    assert R.with_maker(spec("Vehicle", "Toot"), "Bus honking its horn", None, None, None) == "Bus honking its horn"
    assert sense_consistent("Bus", "Toot") and sense_consistent("Computer keyboard", "Typing")


def test_generic_action_gets_its_ontology_maker():
    assert R.with_maker(spec("Tap"), "tap", None, None, None) == "tap"           # Door or a generic tap: no guess
    assert R.with_maker(spec("Dog", "Bark"), "bark", None, None, None) == "a dog making a bark sound"
    assert R.with_maker(spec("Dog", "Bark"), "A dog barking", None, None, None) == "A dog barking"
    assert R.with_maker(spec("Glass", "Chink, clink"), "Two wine glasses clinking", None, None, None) \
        == "Two wine glasses clinking"
    assert R.with_maker(spec("Cough"), "cough", None, None, None) == "a person making a cough sound"
    assert R.with_maker(spec("Bang"), "bang", None, None, None) == "bang"          # no maker: untouched
    assert maker_name("Roaring cats (lions, tigers)") == "lion"


def test_checker_matches_on_the_ontology_label():
    for maker in (False, True):
        config.PICTURE_MAKER = maker
        assert V.ambiguous_entry(spec("Honk"), "a goose honking with its beak open") is None
        assert V.ambiguous_entry(spec("Honk"), "A car honking")["name"] == "horn"      # frames showed a car
        assert V.ambiguous_entry(spec("Vehicle", "Toot"), "Bus honking its horn")["name"] == "horn"
        assert V.ambiguous_entry(spec("Bell", "Ding"), "A bell ringing")["name"] == "bell"
        assert V.ambiguous_entry(spec("Ding"), "a microwave beeping") is None       # Ding is a tone, not a bell
        assert V.ambiguous_entry(spec("Tap"), "a door making a tap sound") is None
    config.PICTURE_MAKER = False


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
