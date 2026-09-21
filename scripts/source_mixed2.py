"""Source a second wave of MIXED candidates (2026-09-21, Adam: "I need at least 30 more
videos that are candidates to mixed ... with clearer mixed style").

"Mixed" under the importance-aware rule (benchmark/gold/tool_template.html, category()):
a needed (off-screen, not obvious) sound of importance >= 2 AND a visible/obvious sound of
importance >= 2 in the same clip. The first wave (scripts/source_targeted.py, mx_) asked
for "one visible source in a busy soundscape" -- the soundscape is texture (traffic, crowd,
waves = importance 1), so most of those clips are seen or unseen under the new rule.

Here every query names an ACTOR DOING AN AUDIBLE ACTION on camera plus a NAMED OFF-SCREEN
EVENT that a deaf viewer would miss (cook frying + phone rings; busker + siren passes;
dog barks at the window + doorbell). Steps per query:
  1. yt-dlp section grab (16-20 s) from the top hits, one clip per video;
  2. the batch-3 quality screen (video+audio, >= 15 s, not silent, not a still, colour);
  3. an AUDIO-ONLY keep rule (BEATs, no visibility model, so it cannot favour clips the
     gate already gets right): at least one drawable sound family above the display bar
     with importance >= 2; tier 1 = two or more distinct such families (the mixed shape),
     tier 2 = one. Tier 1 goes first in the tool. Adam's ticks decide what the clip is.
Survivors land in data/input/benchmark/unsorted/ as m2_<slug>.mp4, the audio families in
benchmark/gold/mixed2_audio.json (for ranking the tagging order), source + licence in
benchmark/sources_mixed2.json.

Usage: python scripts/source_mixed2.py [n_wanted]        (default 30; over-sources ~3x)
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from scripts.source_batch3 import screen, _accept, STAGE, _clip_len
from benchmark.gold.build_tool import importance_of
from src.labels import is_salient_nonspeech, canonical

ARCHIVE = STAGE / "yt_archive.txt"
LOG = ROOT / "benchmark" / "sources_mixed2.json"
AUDIO = ROOT / "benchmark" / "gold" / "mixed2_audio.json"

# (slug, query): visible actor doing something audible + a named off-screen event
QUERIES = [
    ("m2_busker_guitar_siren", "street busker acoustic guitar ambulance siren passing"),
    ("m2_street_drummer_horns", "street drummer buckets performance traffic horns"),
    ("m2_sax_busker_subway", "saxophone busker subway platform train arriving"),
    ("m2_accordion_busker_bells", "accordion street musician church bells ringing"),
    ("m2_dog_window_doorbell", "dog barking at window doorbell rings"),
    ("m2_dog_mailman_truck", "dog barking at mailman delivery truck"),
    ("m2_dog_barks_thunder", "dog barking scared thunder storm outside"),
    ("m2_cat_meow_kitchen", "cat meowing at door kitchen sounds background"),
    ("m2_cooking_phone_rings", "cooking frying pan sizzling phone rings vlog"),
    ("m2_kitchen_doorbell", "kitchen cooking sizzling doorbell rings"),
    ("m2_chopping_market_shouts", "knife chopping vegetables market vendors shouting"),
    ("m2_dishwashing_tv", "washing dishes sink clinking television background"),
    ("m2_blacksmith_church_bell", "blacksmith hammering anvil demonstration church bell"),
    ("m2_blacksmith_fair_crowd", "blacksmith hammering anvil fair crowd horse"),
    ("m2_woodwork_saw_dog", "hand saw woodworking workshop dog barking outside"),
    ("m2_welding_hammer_nearby", "welding sparks workshop hammering nearby"),
    ("m2_pottery_wheel_birds", "pottery wheel throwing workshop birds outside window"),
    ("m2_glass_blowing_workshop", "glass blowing workshop furnace hammer sounds"),
    ("m2_playground_ice_cream", "kids playground swings ice cream truck jingle"),
    ("m2_playground_airplane", "children playing playground airplane flying overhead"),
    ("m2_kids_park_dog_bark", "children laughing park dog barking nearby"),
    ("m2_baby_laugh_dog", "baby laughing high chair dog barking"),
    ("m2_horse_cobbles_tram", "horse hooves cobblestone street tram bell"),
    ("m2_horse_riding_helicopter", "horse riding trail helicopter overhead"),
    ("m2_typing_office_phone", "typing keyboard office phone ringing"),
    ("m2_printer_office_phone", "office printer running phone ringing"),
    ("m2_fountain_pigeons_bells", "fountain plaza pigeons church bells ringing"),
    ("m2_chainsaw_lawnmower", "chainsaw cutting wood neighbour lawnmower"),
    ("m2_lawnmower_siren", "mowing lawn garden ambulance siren passing"),
    ("m2_leaf_blower_dog", "leaf blower yard work dog barking"),
    ("m2_cyclist_bell_siren", "cycling city street bell ringing police siren"),
    ("m2_piano_home_thunder", "playing piano at home thunderstorm outside"),
    ("m2_espresso_cafe_traffic", "barista espresso machine coffee shop street traffic"),
    ("m2_bartender_shaker_crowd", "bartender cocktail shaker bar crowd cheering"),
    ("m2_sewing_machine_radio", "sewing machine workshop radio talking background"),
    ("m2_engine_garage_dog", "car engine revving garage dog barking"),
    ("m2_motorcycle_kick_horns", "motorcycle kick start city traffic horns"),
    ("m2_skateboard_traffic", "skateboarding tricks street traffic passing"),
    ("m2_basketball_court_siren", "basketball dribbling outdoor court siren passing"),
    ("m2_tennis_rally_airplane", "tennis rally court airplane overhead"),
    ("m2_table_tennis_crowd", "table tennis game hall crowd cheering"),
    ("m2_bowling_strike_crowd", "bowling alley strike friends cheering"),
    ("m2_boxing_bag_bell", "boxing gym heavy bag bell rings"),
    ("m2_excavator_jackhammer", "construction excavator digging jackhammer nearby"),
    ("m2_jackhammer_horns", "jackhammer road crew traffic honking"),
    ("m2_fireworks_crowd_dog", "fireworks display crowd cheering dog barking"),
    ("m2_campfire_crickets_owl", "campfire crackling night crickets owl"),
    ("m2_fire_truck_hose", "fire truck siren firefighters hose spraying"),
    ("m2_police_dog_whistle", "police dog training barking whistle"),
    ("m2_sheep_shearing_birds", "sheep shearing farm birds tractor"),
    ("m2_milking_parlour_tractor", "cows milking parlour tractor engine"),
    ("m2_rooster_yard_dog", "rooster crowing farm yard dog barking"),
    ("m2_goat_kids_birds", "goat kids bleating farm birds"),
    ("m2_chickens_tractor", "chickens feeding yard tractor passing"),
    ("m2_ducks_pond_kids", "feeding ducks pond children shouting"),
    ("m2_gulls_pier_horn", "seagulls feeding pier boat horn"),
    ("m2_boat_engine_gulls_bell", "boat engine harbour seagulls bell buoy"),
    ("m2_steam_whistle_crowd", "steam train whistle platform crowd"),
    ("m2_crossing_bells_horn", "level crossing bells train horn passing cars"),
    ("m2_tram_bell_pedestrians", "tram bell ringing street pedestrians"),
    ("m2_ambulance_hospital", "ambulance leaving hospital traffic"),
    ("m2_helicopter_roof_crowd", "helicopter landing hospital roof crowd"),
    ("m2_marching_band_fireworks", "marching band parade fireworks"),
    ("m2_church_bells_pigeons", "church bells ringing tower pigeons flying"),
    ("m2_school_bell_corridor", "school bell ringing kids running corridor"),
    ("m2_football_whistle_vuvuzela", "football match whistle crowd cheering vuvuzela"),
    ("m2_ice_rink_horn", "ice skating rink blades horn"),
    ("m2_gun_range_instructor", "gun range shooting ear protection instructor"),
    ("m2_barber_clippers_traffic", "hairdresser clippers barbershop street traffic"),
    ("m2_dentist_drill_phone", "dentist drill clinic phone ringing"),
    ("m2_vacuum_apartment_dog", "vacuum cleaner apartment dog barking"),
    ("m2_laundromat_door_bell", "laundromat washing machines door bell"),
    ("m2_beekeeper_tractor", "beekeeper hive bees buzzing tractor distant"),
    ("m2_woodpecker_chainsaw", "woodpecker forest chainsaw distant"),
    ("m2_frogs_pond_dogs", "frogs pond night dogs barking distant"),
    ("m2_street_food_wok_horns", "street food wok stir fry traffic horns"),
    ("m2_shoe_shine_street", "shoe shine street brushing traffic"),
    ("m2_car_wash_siren", "washing car driveway hose siren passing"),
    ("m2_axe_chopping_dog", "chopping firewood axe dog barking birds"),
    ("m2_hammer_roof_siren", "roofing hammering nails siren passing"),
    ("m2_bike_repair_shop", "bike repair shop wrench street traffic horns"),
    ("m2_baby_cry_phone", "baby crying phone ringing living room"),
    ("m2_toddler_toys_dog", "toddler playing toys dog barking outside"),
    ("m2_kids_bath_doorbell", "kids bath splashing doorbell rings"),
    ("m2_drum_practice_thunder", "drum practice garage thunder outside"),
    ("m2_violin_practice_dog", "violin practice home dog barking"),
    ("m2_trumpet_practice_siren", "trumpet practice apartment siren outside"),
    ("m2_snow_shovel_plow", "shovelling snow driveway snow plow passing"),
    ("m2_rake_leaves_helicopter", "raking leaves yard helicopter overhead"),
    ("m2_fishing_reel_boat_horn", "fishing reel casting pier boat horn"),
    # wave 2b (more of the same shape)
    ("m2_busker_violin_traffic_horn", "violin busker street car horn honking"),
    ("m2_steel_drum_busker_siren", "steel drum busker siren passing"),
    ("m2_bagpipes_street_bells", "bagpipes street performer church bells"),
    ("m2_dog_fence_barking_truck", "dog barking at fence garbage truck"),
    ("m2_dog_window_siren", "dog howling at siren window"),
    ("m2_cat_toy_doorbell", "cat playing toy doorbell rings reaction"),
    ("m2_parrot_talking_phone", "parrot squawking phone ringing"),
    ("m2_frying_bacon_dog", "frying bacon kitchen dog barking"),
    ("m2_blender_kitchen_doorbell", "blender smoothie kitchen doorbell"),
    ("m2_kettle_whistle_phone", "kettle whistling phone rings kitchen"),
    ("m2_coffee_grinder_siren", "coffee grinder cafe siren outside"),
    ("m2_woodturning_lathe_birds", "wood turning lathe workshop birds"),
    ("m2_forge_hammer_dog", "forge hammering dog barking outside"),
    ("m2_farrier_horseshoe_dog", "farrier shoeing horse dog barking"),
    ("m2_shearing_sheep_dogs", "sheep shearing shed dogs barking"),
    ("m2_horse_trot_church_bell", "horse trotting village church bell"),
    ("m2_cow_bell_tractor", "cows with bells alpine tractor passing"),
    ("m2_pig_feeding_rooster", "pigs feeding trough rooster crowing"),
    ("m2_goose_honking_kids", "geese honking farm children shouting"),
    ("m2_donkey_bray_church", "donkey braying village church bells"),
    ("m2_kids_trampoline_dog", "kids jumping trampoline dog barking"),
    ("m2_baby_rattle_doorbell", "baby playing rattle doorbell rings"),
    ("m2_toddler_piano_phone", "toddler banging piano phone ringing"),
    ("m2_kids_basketball_siren", "kids playing basketball driveway siren"),
    ("m2_skipping_rope_ice_cream", "kids skipping rope street ice cream truck"),
    ("m2_scooter_kids_helicopter", "kids scooter park helicopter"),
    ("m2_typing_thunder", "typing laptop desk thunderstorm outside window"),
    ("m2_sewing_dog_barking", "sewing machine dog barking"),
    ("m2_knitting_needles_phone", "knitting needles clicking phone ringing"),
    ("m2_dishes_kitchen_siren", "washing dishes kitchen window siren outside"),
    ("m2_vacuum_doorbell", "vacuuming doorbell rings dog"),
    ("m2_drill_diy_dog", "drilling wall diy dog barking"),
    ("m2_hammer_nails_thunder", "hammering nails deck thunder"),
    ("m2_sanding_workshop_siren", "sanding workshop siren outside"),
    ("m2_mower_ride_helicopter", "riding lawn mower helicopter overhead"),
    ("m2_hedge_trimmer_church", "hedge trimmer garden church bells"),
    ("m2_pressure_washer_dog", "pressure washing patio dog barking"),
    ("m2_car_horn_wedding_bells", "wedding car horns church bells"),
    ("m2_motorbike_wheelie_siren", "motorcycle revving street police siren"),
    ("m2_bus_doors_beep_siren", "bus door beeping stop ambulance siren"),
    ("m2_train_doors_whistle", "train doors closing whistle platform"),
    ("m2_tram_bell_church", "tram passing bell church bells square"),
    ("m2_bike_bell_tram", "bicycle bell ringing tram passing"),
    ("m2_ambulance_dog_bark", "ambulance siren dog barking street"),
    ("m2_fire_engine_kids", "fire engine siren children waving"),
    ("m2_street_market_scooter_horn", "street market vendor scooter horn"),
    ("m2_fish_market_auction_gulls", "fish market auction shouting seagulls"),
    ("m2_butcher_chopping_traffic", "butcher chopping cleaver street traffic"),
    ("m2_cobbler_hammer_street", "cobbler hammering shoes street sounds"),
    ("m2_tailor_scissors_radio", "tailor cutting fabric scissors radio"),
    ("m2_barista_steam_dog", "barista steaming milk dog barking cafe"),
    ("m2_bartender_ice_glass_break", "bartender ice bucket glass breaks bar"),
    ("m2_restaurant_kitchen_phone", "restaurant kitchen pans phone ringing"),
    ("m2_ping_pong_dog", "ping pong garage dog barking"),
    ("m2_golf_swing_airplane", "golf swing driving range airplane"),
    ("m2_archery_range_birds", "archery range arrows birds"),
    ("m2_baseball_bat_crowd", "baseball batting cage crowd cheering"),
    ("m2_hockey_stick_horn", "hockey practice puck horn"),
    ("m2_gym_weights_phone", "gym weights dropping phone ringing"),
    ("m2_horse_jumping_crowd", "show jumping horse crowd applause"),
    ("m2_rowing_oars_horn", "rowing boat oars river boat horn"),
    ("m2_kayak_paddle_helicopter", "kayak paddling river helicopter"),
    ("m2_swimming_pool_whistle", "swimming lesson pool whistle lifeguard"),
    ("m2_axe_throw_crowd", "axe throwing crowd cheering"),
    ("m2_drum_circle_siren", "drum circle park siren passing"),
    ("m2_church_organ_bells", "church organ practice bells tower"),
    ("m2_school_recorder_bell", "school children recorder practice bell rings"),
    ("m2_orchestra_rehearsal_phone", "orchestra rehearsal phone rings"),
    ("m2_dj_street_party_horns", "dj street party car horns"),
    ("m2_fireworks_kids_dog", "fireworks backyard kids screaming dog barking"),
    ("m2_bonfire_crowd_siren", "bonfire night crowd siren"),
    ("m2_snowball_fight_plow", "snowball fight kids snow plow"),
    ("m2_sledding_kids_dog", "kids sledding hill dog barking"),
    ("m2_rain_umbrella_thunder", "walking rain umbrella thunder city"),
    ("m2_car_wash_tunnel_horn", "car wash tunnel inside horn"),
    ("m2_gas_station_pump_siren", "gas station pumping fuel siren passing"),
    ("m2_mechanic_impact_wrench_dog", "mechanic impact wrench garage dog"),
    ("m2_tire_change_traffic_horn", "roadside tire change traffic horns"),
    ("m2_drive_thru_speaker_siren", "drive thru speaker order siren"),
    ("m2_atm_beeps_street", "atm beeping street traffic horns"),
    ("m2_arcade_claw_crowd", "arcade claw machine crowd cheering"),
    ("m2_bowling_kids_thunder", "bowling kids cheering thunder outside"),
    ("m2_zoo_lion_roar_kids", "zoo lion roaring children screaming"),
    ("m2_monkey_enclosure_phone", "zoo monkeys screeching phone ringing"),
    ("m2_elephant_trumpet_crowd", "elephant trumpeting zoo crowd"),
    ("m2_seal_show_clapping", "sea lion show barking applause"),
    ("m2_aviary_parrots_kids", "aviary parrots squawking kids"),
    ("m2_dog_show_applause", "dog agility show barking applause"),
    ("m2_cat_cafe_door_chime", "cat cafe meowing door chime"),
]


DEEP = "--deep" in sys.argv          # second pass: hits 7-12 of every query
AUDIO_BAR = 0.25                      # keep bar for the audio rule (ranking only; the VLM screen and Adam decide)


def _download(name: str, query: str) -> Path | None:
    raw = STAGE / f"{name}.mp4"
    raw.unlink(missing_ok=True)
    sec = _clip_len()
    start = random.choice((45, 60, 90))
    # one call: the first of the top 6 hits that is long enough and not in the archive
    # (ytsearchN + --playlist-items N kept returning hit 1 with this yt-dlp build)
    for _ in (1,):
        subprocess.run(
            ["yt-dlp", f"ytsearch{12 if DEEP else 6}:{query}", "--max-downloads", "1"]
            + (["--playlist-items", "7:12"] if DEEP else []) + [
             "--download-sections", f"*{start}-{start + sec}",
             "-f", "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/b",
             "--merge-output-format", "mp4",
             "--force-keyframes-at-cuts",
             "--match-filter", f"duration>{start + 60} & duration<7200",
             "--download-archive", str(ARCHIVE),
             "--print-to-file", "%(webpage_url)s|%(title)s", str(STAGE / f"{name}.meta"),
             "--no-warnings", "-o", str(raw)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=420)
        if raw.exists():
            return raw
    return None


def audio_families(video: Path) -> list[dict]:
    """drawable sound families above the display bar with importance >= 2 (BEATs, audio only)"""
    from src.stage1_audio_extraction import extract_audio
    from src.stage4_audio_event_detection import detect_events
    config.LABEL_FILTER = "depictable"
    with tempfile.TemporaryDirectory() as td:
        media = extract_audio(video, Path(td) / "a.wav", config.SAMPLE_RATE)
        events = detect_events(Path(media.wav_path), threshold=config.AED_THRESHOLD,
                               min_dur=config.AED_MIN_DUR, device="cuda", model="beats")
    fams: dict[str, dict] = {}
    for e in events:
        if e.confidence < AUDIO_BAR or not is_salient_nonspeech(e.label):
            continue
        fam = canonical(e.label)
        if importance_of(e.label.lower(), fam) < 2:
            continue
        f = fams.setdefault(fam, {"family": fam, "detail": e.label, "conf": 0.0, "start": e.start, "end": e.end})
        if e.confidence > f["conf"]:
            f.update(detail=e.label, conf=round(float(e.confidence), 2), start=round(e.start, 1), end=round(e.end, 1))
    return sorted(fams.values(), key=lambda f: -f["conf"])


def main(want_n: int) -> None:
    STAGE.mkdir(parents=True, exist_ok=True)
    audio = json.loads(AUDIO.read_text(encoding="utf-8")) if AUDIO.exists() else {}
    queries = QUERIES[:]
    random.shuffle(queries)
    got = 0
    for name, query in queries:
        if got >= want_n:
            break
        if DEEP:
            name = name + "_b"
        if any((ROOT / "data" / "input" / "benchmark" / d / f"{name}.mp4").exists()
               for d in ("unsorted", "mixed", "seen_ambient", "unseen_ambient", "no_ambient", "_dropped", "_bad")):
            continue
        print(f"  {name}: {query}", flush=True)
        raw = _download(name, query)
        if raw is None:
            print("    skip (no download)", flush=True)
            continue
        why = screen(raw)
        if why:
            print(f"    reject ({why})", flush=True)
            raw.unlink(missing_ok=True)
            continue
        try:
            fams = audio_families(raw)
        except Exception as e:
            print(f"    skip (audio analysis {type(e).__name__}: {e})", flush=True)
            raw.unlink(missing_ok=True)
            continue
        if not fams:
            print("    drop (no drawable sound of importance >= 2)", flush=True)
            raw.unlink(missing_ok=True)
            continue
        meta_f = STAGE / f"{name}.meta"
        url, title = "", ""
        if meta_f.exists():
            parts = meta_f.read_text(encoding="utf-8").strip().split("|", 1)
            url, title = parts[0], (parts[1] if len(parts) > 1 else "")
        import scripts.source_batch3 as b3
        b3.LOG = LOG          # own source log
        if _accept(raw, name, {"source": "youtube", "url": url, "title": title,
                               "targeted": "mixed2", "license": "research use; not redistributed"}):
            got += 1
            audio[name] = {"tier": 1 if len(fams) >= 2 else 2, "families": fams}
            AUDIO.write_text(json.dumps(audio, indent=1, ensure_ascii=False), encoding="utf-8")
            print(f"    KEEP {got}/{want_n} tier {1 if len(fams) >= 2 else 2}: {[f['family'] for f in fams]}", flush=True)
    print(f"done: {got} kept")


if __name__ == "__main__":
    main(int([a for a in sys.argv[1:] if not a.startswith("--")][0]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else 30)
