"""Source clips for a SPECIFIC benchmark category (mixed / unseen_ambient).

Earlier waves sourced by scene type and accepted whatever came out, which
over-produced no_ambient and seen_ambient. Here we over-source, then run the
real analysis (PANNs + CLIP + gate, i.e. benchmark/suggest.py) on each candidate
and KEEP ONLY the ones predicted to be the category we still need; the rest are
deleted immediately so nothing lands in the tagging queue that Adam does not need.

"mixed" needs a scene with ONE obvious on-screen sound source inside a busy
soundscape (a visible dog barking on a traffic street, a fountain with birds
overhead) -- the queries below are chosen for that shape.

Usage:
    python scripts/source_targeted.py mixed 40
    python scripts/source_targeted.py unseen_ambient 30
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.source_batch3 import screen, _accept, STAGE, _clip_len

ARCHIVE = STAGE / "yt_archive.txt"
UNSORTED = ROOT / "data" / "input" / "benchmark" / "unsorted"
SUGG = ROOT / "benchmark" / "suggestions.json"

# --- MIXED: one prominent sound source ON screen, others off ------------------
MIXED_Q = [
    ("mx_dog_walk_street", "walking dog on busy street pov traffic"),
    ("mx_dog_park_traffic", "dog park near road barking traffic"),
    ("mx_fountain_birds", "city fountain square birds pigeons people"),
    ("mx_busker_traffic", "street performer busking traffic passing behind"),
    ("mx_market_stall_frying", "street food stall frying cooking market crowd"),
    ("mx_train_platform_wait", "standing on train platform announcement train arrives"),
    ("mx_tram_stop_street", "tram stop street waiting traffic"),
    ("mx_construction_street", "construction work next to busy road"),
    ("mx_roadworks_jackhammer", "jackhammer roadworks pedestrians passing"),
    ("mx_chainsaw_garden", "chainsaw cutting tree garden birds"),
    ("mx_lawnmower_birds", "mowing lawn garden birds neighbourhood"),
    ("mx_waterfall_visitors", "waterfall viewpoint visitors talking birds"),
    ("mx_river_ducks", "river bank ducks quacking city bridge"),
    ("mx_beach_seagulls_kids", "beach seagulls children playing waves"),
    ("mx_harbor_boat_engine", "harbour boat engine seagulls dock"),
    ("mx_playground_road", "playground next to road children traffic"),
    ("mx_school_yard_street", "school yard break children street"),
    ("mx_cafe_terrace_traffic", "cafe terrace street traffic passing people talking"),
    ("mx_farmer_tractor_birds", "tractor field work birds countryside"),
    ("mx_horse_riding_road", "horse riding along road cars passing"),
    ("mx_bike_ride_city", "cycling city street pov traffic bell"),
    ("mx_skatepark_street", "skatepark near street traffic skating"),
    ("mx_basketball_outdoor_city", "outdoor basketball court city traffic game"),
    ("mx_dog_beach_waves", "dogs running beach waves barking"),
    ("mx_chickens_yard_road", "chickens yard village road vehicles"),
    ("mx_sheep_road_crossing", "sheep crossing road farmer cars"),
    ("mx_fishing_pier_gulls", "fishing off pier seagulls boats"),
    ("mx_train_crossing_wait", "waiting at railroad crossing train passes cars"),
    ("mx_bus_stop_wait", "waiting bus stop street traffic bus arrives"),
    ("mx_airport_spotting", "plane spotting airport fence aircraft landing"),
    ("mx_helicopter_landing_crowd", "helicopter landing people watching"),
    ("mx_parade_street_crowd", "parade passing street crowd cheering"),
    ("mx_protest_traffic", "protest march street traffic horns"),
    ("mx_fire_truck_scene", "fire truck at scene firefighters street"),
    ("mx_ambulance_scene_street", "ambulance at scene street bystanders"),
    ("mx_church_square_bells", "church square bells people walking"),
    ("mx_temple_courtyard", "temple courtyard visitors bells birds"),
    ("mx_zoo_enclosure_visitors", "zoo enclosure animals visitors talking"),
    ("mx_aquarium_tank_crowd", "aquarium tank visitors children"),
    ("mx_pool_kids_traffic", "outdoor swimming pool children street"),
    ("mx_bbq_garden_neighbours", "garden barbecue talking neighbourhood sounds"),
    ("mx_carwash_street", "washing car driveway street traffic"),
    ("mx_delivery_truck_street", "delivery truck unloading street city"),
    ("mx_garbage_collection_street", "garbage truck collecting street morning"),
    ("mx_street_sweeper_pedestrians", "street sweeper machine pedestrians"),
    ("mx_food_truck_queue", "food truck queue street cooking traffic"),
    # wave 2
    ("mx_fish_market_gulls", "fish market morning seagulls vendors"),
    ("mx_ferry_dock_cars", "ferry dock cars loading gulls"),
    ("mx_level_crossing_town", "level crossing town traffic barrier train"),
    ("mx_bus_terminal_city", "bus terminal buses arriving people"),
    ("mx_taxi_rank_street", "taxi rank street city waiting"),
    ("mx_pedestrian_crossing_beep", "pedestrian crossing beeping traffic waiting"),
    ("mx_scooter_repair_street", "scooter repair shop street traffic"),
    ("mx_welding_street_shop", "welding workshop open to street"),
    ("mx_carpenter_yard_birds", "carpenter working outdoors birds"),
    ("mx_stone_mason_yard", "stonemason working yard sounds"),
    ("mx_potter_market_stall", "potter working market visitors"),
    ("mx_blacksmith_fair", "blacksmith demonstration fair visitors"),
    ("mx_farm_visit_animals", "farm visit children animals tractor"),
    ("mx_stable_yard_horses", "stable yard horses people working"),
    ("mx_kennels_dogs", "dog kennels barking staff walking"),
    ("mx_vet_clinic_waiting", "vet clinic waiting room dogs"),
    ("mx_pet_market_birds", "bird market cages vendors"),
    ("mx_duck_pond_feeding", "feeding ducks pond park people"),
    ("mx_swan_river_boats", "swans river boats passing"),
    ("mx_canal_lock_operation", "canal lock operating boat water"),
    ("mx_watermill_visitors", "watermill working visitors water"),
    ("mx_windmill_visitors", "windmill turning visitors sounds"),
    ("mx_steam_fair_engines", "steam fair traction engines crowd"),
    ("mx_vintage_car_show", "classic car show engines crowd"),
    ("mx_motorbike_meet", "motorcycle meet up engines people"),
    ("mx_air_show_crowd", "air show aircraft crowd watching"),
    ("mx_marching_band_street", "marching band street parade spectators"),
    ("mx_fire_drill_street", "fire drill building alarm people outside"),
    ("mx_road_race_spectators", "road race runners spectators cheering street"),
    ("mx_cycling_race_roadside", "cycling race roadside spectators bikes"),
    ("mx_soccer_training_pitch", "football training pitch whistle players"),
    ("mx_tennis_club_courts", "tennis club courts play ambience"),
    ("mx_golf_course_players", "golf course players shots birds"),
    ("mx_fishing_boat_deck", "fishing boat deck working gulls engine"),
    ("mx_beach_lifeguard_whistle", "lifeguard whistle beach crowd waves"),
]

# --- UNSEEN: rich off-screen soundscape, camera pointed elsewhere -------------
UNSEEN_Q = [
    ("us_window_view_street", "filming out window street sounds below"),
    ("us_balcony_city_night", "balcony view city night sounds traffic"),
    ("us_rooftop_terrace_city", "rooftop terrace city ambience below"),
    ("us_indoor_window_rain", "indoor window rain traffic outside"),
    ("us_courtyard_from_above", "courtyard from above people sounds"),
    ("us_alley_quiet_city", "quiet alley city sounds around corner"),
    ("us_park_bench_city", "sitting park bench city sounds around"),
    ("us_forest_edge_road", "forest path near road distant traffic"),
    ("us_hide_birdwatching", "birdwatching hide birds calling around"),
    ("us_tent_morning_forest", "morning inside tent forest birds outside"),
    ("us_cabin_storm_inside", "inside cabin storm wind rain outside"),
    ("us_car_interior_traffic", "sitting in parked car street sounds outside"),
    ("us_bus_interior_city", "inside bus city sounds outside window"),
    ("us_train_interior_window", "inside train window countryside sounds"),
    ("us_ferry_deck_engine", "ferry deck sea engine below"),
    ("us_stairwell_building", "stairwell building echo sounds"),
    ("us_parking_garage_sounds", "parking garage sounds cars echo"),
    ("us_tunnel_walk_traffic", "walking pedestrian tunnel traffic above"),
    ("us_underpass_train_above", "under railway bridge train passing above"),
    ("us_museum_hall_visitors", "museum hall quiet visitors distant"),
    ("us_library_reading_room", "library reading room ambience"),
    ("us_empty_stadium_outside", "outside stadium crowd noise inside"),
    ("us_backstage_event", "backstage event crowd noise beyond"),
    ("us_kitchen_door_restaurant", "restaurant dining room kitchen sounds behind"),
    ("us_hotel_room_street", "hotel room window city street noise"),
    ("us_office_window_city", "office window city sounds outside"),
    ("us_hospital_corridor", "hospital corridor sounds ambience"),
    ("us_night_village_sounds", "village night sounds dogs distant"),
    ("us_desert_camp_night", "desert camp night wind sounds"),
    ("us_lake_dawn_mist", "lake dawn mist birds distant sounds"),
    ("us_field_far_machinery", "open field distant farm machinery"),
    ("us_hillside_valley_sounds", "hillside overlooking valley village sounds"),
    ("us_snow_forest_quiet", "snow forest walk quiet distant sounds"),
    ("us_cave_entrance_sounds", "cave entrance dripping echo outside"),
    ("us_marina_night_water", "marina night water boats sounds"),
    ("us_pier_night_waves", "pier at night waves sounds"),
    # wave 2
    ("us_attic_rain_roof", "attic room rain on roof sounds"),
    ("us_basement_pipes", "basement boiler room pipes sounds"),
    ("us_garage_inside_street", "inside garage door open street sounds"),
    ("us_shed_garden_birds", "garden shed inside birds outside"),
    ("us_greenhouse_rain", "greenhouse rain on glass"),
    ("us_barn_inside_animals", "inside barn animals outside sounds"),
    ("us_bridge_under_river", "under bridge river traffic above"),
    ("us_subway_platform_wait", "subway platform waiting distant train"),
    ("us_escalator_hall_echo", "station hall escalators echo announcements"),
    ("us_church_interior_street", "inside church quiet street outside"),
    ("us_mosque_interior_city", "inside mosque courtyard city sounds"),
    ("us_shop_interior_street", "small shop interior street outside door"),
    ("us_barber_interior_street", "barbershop interior street sounds outside"),
    ("us_laundry_room_machines", "laundry room machines running ambience"),
    ("us_server_room_hum", "server room fans hum walkthrough"),
    ("us_factory_corridor", "factory corridor machines beyond doors"),
    ("us_backyard_neighbours", "backyard neighbours sounds fence"),
    ("us_apartment_courtyard", "apartment courtyard echo residents"),
    ("us_stair_landing_city", "stair landing window city noise"),
    ("us_rooftop_helipad_city", "rooftop city wind traffic below"),
    ("us_forest_clearing_distant", "forest clearing distant chainsaw birds"),
    ("us_meadow_distant_road", "meadow grass distant road cars"),
    ("us_orchard_distant_tractor", "orchard trees distant tractor birds"),
    ("us_riverbank_hidden", "riverbank reeds hidden water birds"),
    ("us_dune_behind_beach", "sand dunes behind beach waves distant"),
]


def _download(name: str, query: str) -> Path | None:
    raw = STAGE / f"tg_{name}.mp4"
    raw.unlink(missing_ok=True)
    sec = _clip_len()
    for rank in (1, 2):
        subprocess.run(
            ["yt-dlp", f"ytsearch{rank}:{query}", "--playlist-items", str(rank),
             "--download-sections", f"*120-{120 + sec}",
             "-f", "mp4[height<=720]/best[height<=720]/best",
             "--force-keyframes-at-cuts",
             "--match-filter", "duration>180 & duration<7200",
             "--download-archive", str(ARCHIVE),
             "--print-to-file", "%(webpage_url)s|%(title)s", str(STAGE / f"tg_{name}.meta"),
             "--no-warnings", "-o", str(raw)],
            capture_output=True, text=True, timeout=420)
        if raw.exists():
            return raw
    return None


ACCEPT = {"mixed", "unseen_ambient", "seen_ambient"}


def main(want_cat: str, want_n: int):
    from benchmark.suggest import suggest_clip
    STAGE.mkdir(parents=True, exist_ok=True)
    queries = (MIXED_Q if want_cat == "mixed" else UNSEEN_Q)[:]
    random.shuffle(queries)
    sug = json.loads(SUGG.read_text(encoding="utf-8")) if SUGG.exists() else {}
    got, tried = 0, 0
    for name, query in queries:
        if got >= want_n:
            break
        tried += 1
        print(f"  [{want_cat}] {name}: {query}", flush=True)
        raw = _download(name, query)
        if raw is None:
            print("    skip (no download)")
            continue
        why = screen(raw)
        if why:
            print(f"    reject ({why})")
            raw.unlink(missing_ok=True)
            continue
        # the decisive step: does the real gate think this is the category we need?
        try:
            pred = suggest_clip(raw)
        except Exception as e:
            print(f"    skip (analysis {type(e).__name__})")
            raw.unlink(missing_ok=True)
            continue
        # Inclusion rule: keep anything with a DETECTED non-speech sound; reject
        # only "no_ambient" (nothing to gate on). We deliberately do NOT require the
        # gate to predict the target category -- selecting clips by the model's own
        # prediction would bias the benchmark toward cases the gate already gets
        # right and inflate the reported accuracy. Adam's tag is the ground truth.
        if pred["suggest"] not in ACCEPT:
            print(f"    drop (predicted {pred['suggest']})")
            raw.unlink(missing_ok=True)
            continue
        meta_f = STAGE / f"tg_{name}.meta"
        url, title = "", ""
        if meta_f.exists():
            parts = meta_f.read_text(encoding="utf-8").strip().split("|", 1)
            url = parts[0]
            title = parts[1] if len(parts) > 1 else ""
        if _accept(raw, name, {"source": "youtube", "url": url, "title": title,
                               "targeted": want_cat,
                               "license": "research use; not redistributed"}):
            sug[f"unsorted/{name}.mp4"] = pred
            SUGG.write_text(json.dumps(sug, indent=2, ensure_ascii=False), encoding="utf-8")
            got += 1
            print(f"    KEEP ({got}/{want_n}) [{pred['suggest']}] -- {pred['reason']}")
    print(f"\n{want_cat}: kept {got} of {tried} tried")


if __name__ == "__main__":
    cat = sys.argv[1] if len(sys.argv) > 1 else "mixed"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    main(cat, n)
