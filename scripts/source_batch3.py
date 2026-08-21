"""Batch-3 sourcing: ~150+ diverse unseen/mixed candidates for the benchmark.

Three sources, per-source quotas so no single origin dominates:
  yt   — yt-dlp section-grabs (~28 s) from top search hits across ~70 distinct
         scene queries (one clip per video, --download-archive prevents repeats)
  wm   — Wikimedia Commons API video search across varied terms (CC), whole
         file if small, middle 28 s cut locally
  ia   — Internet Archive color travelogues/home-movies, mp4 derivative
         segment-grab over HTTP (works on IA mp4s)

Every candidate passes the quality screen learned from Adam's BAD tags:
video+audio streams, >=15 s, mean_volume > -50 dB, not a still image, not B&W.
Survivors land in data/input/benchmark/unsorted/ as b3_<theme>.mp4 and are
logged with source+license in benchmark/sources_batch3.json.

Usage: python scripts/source_batch3.py [yt|wm|ia|all]
"""
from __future__ import annotations

import json
import random
import re
import statistics
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UNSORTED = ROOT / "data" / "input" / "benchmark" / "unsorted"
STAGE = ROOT / "data" / "work" / "batch3_staging"
LOG = ROOT / "benchmark" / "sources_batch3.json"
ARCHIVE = STAGE / "yt_archive.txt"
HDRS = {"User-Agent": "MscFinalProject/0.1 (academic research; benchmark curation)"}
CLIP_SEC = 28          # waves 1-3 used a fixed cut


def _clip_len() -> int:
    """Waves 4+: varied clip lengths so the benchmark isn't uniform.
    16 s floor keeps a keyframe-imprecise cut above the >=15 s screen."""
    return random.randint(16, 25)

# ---------------------------------------------------------------- quality screen
def _probe(f: Path):
    try:
        o = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                            "format=duration:stream=codec_type", "-of", "json", str(f)],
                           capture_output=True, text=True, timeout=60).stdout
        d = json.loads(o or "{}")
        dur = float(d.get("format", {}).get("duration", 0) or 0)
        types = [s.get("codec_type") for s in d.get("streams", [])]
        return dur, "video" in types, "audio" in types
    except Exception:
        return 0, False, False


def _meanvol(f: Path):
    try:
        o = subprocess.run(["ffmpeg", "-i", str(f), "-af", "volumedetect", "-f", "null", "-"],
                           capture_output=True, text=True, timeout=120).stderr or ""
    except Exception:
        return -99.0
    m = re.search(r"mean_volume:\s*(-?[\d.]+) dB", o)
    return float(m.group(1)) if m else -99.0


def _frame(f: Path, t: float, td: str, tag: str):
    from PIL import Image
    fp = Path(td) / f"{tag}.jpg"
    subprocess.run(["ffmpeg", "-y", "-ss", f"{t:.1f}", "-i", str(f), "-frames:v", "1",
                    "-vf", "scale=160:-1", str(fp)], capture_output=True, timeout=60)
    return Image.open(fp) if fp.exists() else None


def screen(f: Path) -> str | None:
    """Return a rejection reason, or None if the clip is good."""
    from PIL import ImageChops
    dur, hasv, hasa = _probe(f)
    if not hasv:
        return "no video"
    if not hasa:
        return "no audio"
    if dur < 15:
        return f"short {dur:.0f}s"
    if _meanvol(f) < -50:
        return "near-silent"
    with tempfile.TemporaryDirectory() as td:
        a, b = _frame(f, dur * 0.3, td, "a"), _frame(f, dur * 0.7, td, "b")
        if a and b and a.size == b.size:
            if statistics.mean(ImageChops.difference(
                    a.convert("L"), b.convert("L")).getdata()) < 2.0:
                return "still image"
        if a and statistics.mean(a.convert("HSV").split()[1].getdata()) < 14:
            return "black&white"
    return None


# ---------------------------------------------------------------- helpers
def _log(entry: dict):
    log = json.loads(LOG.read_text(encoding="utf-8")) if LOG.exists() else []
    log.append(entry)
    LOG.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")


def _accept(tmp: Path, name: str, meta: dict) -> bool:
    try:
        why = screen(tmp)
    except Exception as e:
        why = f"screen error {type(e).__name__}"
    if why:
        print(f"    reject {name}: {why}")
        tmp.unlink(missing_ok=True)
        return False
    dest = UNSORTED / f"{name}.mp4"
    if dest.exists():
        tmp.unlink(missing_ok=True)
        return False
    if tmp.suffix != ".mp4":
        subprocess.run(["ffmpeg", "-y", "-i", str(tmp), "-c:v", "libx264", "-preset",
                        "veryfast", "-crf", "26", "-c:a", "aac", str(dest)],
                       capture_output=True, timeout=300)
        tmp.unlink(missing_ok=True)
    else:
        tmp.replace(dest)
    if not dest.exists():
        return False
    _log({"file": dest.name, **meta})
    print(f"    OK  {dest.name}")
    return True


def _cut_middle(src: Path, out: Path, sec: int | None = None):
    sec = sec or _clip_len()
    dur, _, _ = _probe(src)
    start = max(0.0, dur / 2 - sec / 2)
    subprocess.run(["ffmpeg", "-y", "-ss", f"{start:.1f}", "-i", str(src),
                    "-t", str(sec), "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "26", "-c:a", "aac", str(out)],
                   capture_output=True, timeout=300)


# ---------------------------------------------------------------- source: YouTube
YT_QUERIES = [
    # street life / walks — many countries
    ("walk_lagos", "lagos nigeria street walk market sounds"),
    ("walk_hanoi", "hanoi old quarter walking tour motorbikes"),
    ("walk_cairo", "cairo street walk khan el khalili"),
    ("walk_mexicocity", "mexico city centro walking tour street"),
    ("walk_mumbai_rain", "mumbai monsoon rain street walk"),
    ("walk_seoul_night", "seoul night market street walk"),
    ("walk_naples", "naples italy narrow streets walking scooters"),
    ("walk_havana", "havana cuba street walk old cars"),
    ("walk_athens", "athens plaka walking tour street musicians"),
    ("walk_saopaulo", "sao paulo downtown street walk"),
    # markets / bazaars
    ("market_marrakech", "marrakech souk walking tour"),
    ("market_taipei_night", "taipei night market food walk"),
    ("market_istanbul_spice", "istanbul spice bazaar walk"),
    ("market_fish_tokyo", "tsukiji fish market walk"),
    ("market_floating", "bangkok floating market boat ride"),
    ("market_delhi", "delhi chandni chowk market walk"),
    # water / harbors / boats
    ("harbor_fishing", "fishing harbor boats seagulls morning"),
    ("ferry_ride", "ferry ride deck sea sound"),
    ("kayak_river", "kayaking river rapids sounds"),
    ("beach_boardwalk", "beach boardwalk summer crowd walk"),
    ("canal_amsterdam", "amsterdam canal boat ride"),
    ("venice_gondola", "venice gondola ride sounds ambient"),
    # nature / rural
    ("farm_sheep", "sheep farm ambient sounds new zealand"),
    ("jungle_hike", "jungle hike borneo sounds birds"),
    ("waterfall_hike", "waterfall hike trail sound"),
    ("forest_rain", "rain forest walk sound"),
    ("desert_camel", "desert camel ride wind"),
    ("savanna_safari", "safari game drive sounds animals"),
    ("mountain_stream", "mountain stream hiking sounds"),
    ("rice_field", "rice field village sounds vietnam"),
    # weather
    ("storm_street", "thunderstorm city street heavy rain"),
    ("snowstorm_walk", "snowstorm blizzard walk city"),
    ("wind_coast", "strong wind coastal cliff walk"),
    ("hail_storm", "hailstorm on street video sound"),
    # transport (non-train-heavy)
    ("airport_terminal", "airport terminal walk announcements"),
    ("tram_lisbon", "lisbon tram 28 ride sounds"),
    ("tuktuk_ride", "tuk tuk ride bangkok traffic"),
    ("cablecar_ride", "cable car ride sounds mountain"),
    ("motorbike_traffic", "vietnam motorbike traffic crossing street"),
    ("bus_ride_city", "city bus ride ambient sounds"),
    # events / people
    ("carnival_parade", "carnival parade street drums brazil"),
    ("festival_lantern", "lantern festival crowd walk"),
    ("street_performer", "street performer crowd watching plaza"),
    ("stadium_entrance", "football stadium entrance crowd chants outside"),
    ("protest_march", "protest march chanting street"),
    ("wedding_street", "street wedding procession music india"),
    ("church_square", "church square bells pigeons people"),
    ("playground_park", "playground park children playing sounds"),
    ("basketball_court", "outdoor basketball court pickup game sounds"),
    ("skatepark", "skatepark sounds skating ambient"),
    # work / industry
    ("construction_site", "construction site sounds excavator"),
    ("workshop_blacksmith", "blacksmith workshop hammering"),
    ("sawmill_wood", "sawmill woodworking sounds"),
    ("car_wash", "car wash from inside sounds"),
    ("barbershop", "barbershop ambient clippers sounds"),
    ("kitchen_restaurant", "restaurant kitchen sounds cooking service"),
    ("farm_tractor", "tractor plowing field sounds"),
    ("port_cranes", "container port cranes loading sounds"),
    # indoor / misc
    ("cafe_terrace", "cafe terrace street ambient people"),
    ("library_hall", "library hall quiet ambience pages"),
    ("arcade_games", "arcade game hall sounds walk"),
    ("bowling_alley", "bowling alley sounds strikes"),
    ("swimming_pool", "indoor swimming pool echo sounds"),
    ("zoo_walk", "zoo walking tour animal sounds"),
    ("aquarium_walk", "aquarium walk ambient"),
    ("night_crickets", "village night crickets dogs distant"),
    ("campfire_night", "campfire night sounds crackling talking"),
    ("fireworks_festival", "fireworks festival crowd watching"),
    ("marina_masts", "marina sailboat masts clinking wind"),
    ("food_court", "food court mall lunchtime ambience"),
    # wave 2 — fresh themes for the resume pass
    ("walk_tokyo_rain", "tokyo rain night walk umbrella"),
    ("walk_ny_chinatown", "new york chinatown street walk"),
    ("walk_paris_morning", "paris morning walk cafe street"),
    ("walk_dubai_souk", "dubai gold souk walk"),
    ("walk_moscow_winter", "moscow winter street walk snow"),
    ("walk_buenosaires", "buenos aires san telmo street walk"),
    ("walk_manila", "manila street walk jeepney"),
    ("walk_addis", "addis ababa street walk market"),
    ("horse_ranch", "horse ranch stable sounds"),
    ("cow_farm", "dairy farm cows milking sounds"),
    ("chicken_coop", "chicken farm rooster sounds village"),
    ("bee_apiary", "beekeeper apiary bees buzzing"),
    ("frog_pond", "frog pond night chorus video"),
    ("bat_cave", "cave tour dripping echo sounds"),
    ("geyser_hotspring", "geyser eruption yellowstone sound"),
    ("glacier_hike", "glacier hike crunching ice wind"),
    ("volcano_tour", "volcano crater tour steam sounds"),
    ("sand_dunes", "sand dunes wind quad bikes"),
    ("vineyard_harvest", "vineyard grape harvest sounds"),
    ("olive_press", "olive oil mill press sounds"),
    ("pottery_wheel", "pottery workshop wheel sounds"),
    ("glass_blowing", "glass blowing workshop furnace"),
    ("printing_press", "printing press machine sounds"),
    ("textile_loom", "weaving loom workshop sounds"),
    ("laundromat", "laundromat machines ambience"),
    ("gym_weights", "gym weights dropping sounds ambience"),
    ("boxing_gym", "boxing gym training bags sounds"),
    ("ice_rink", "ice skating rink sounds ambience"),
    ("tennis_court", "tennis match court sounds amateur"),
    ("golf_course", "golf course driving range sounds"),
    ("horse_race", "horse race track sounds crowd"),
    ("regatta_sailing", "sailing regatta start sounds"),
    ("drone_field", "rc drone flying field sounds"),
    ("gokart_track", "go kart track sounds racing"),
    ("paintball_field", "paintball field game sounds"),
    ("climbing_gym", "climbing gym ambience sounds"),
    ("night_market_food", "street food frying sizzling night market"),
    ("butcher_market", "butcher meat market sounds"),
    ("flower_market", "flower market morning sounds"),
    ("auction_livestock", "livestock auction auctioneer sounds"),
    # wave 3 — final fill to 150
    ("medina_fes", "fes morocco medina walk donkeys"),
    ("bazaar_tehran", "tehran grand bazaar walk"),
    ("oldtown_prague", "prague old town walk tourists"),
    ("oldtown_krakow", "krakow main square walk pigeons"),
    ("township_capetown", "cape town township street walk"),
    ("favela_rio", "rio favela street walk sounds"),
    ("walk_kyoto_gion", "kyoto gion evening walk"),
    ("walk_lisbon_alfama", "lisbon alfama walk fado streets"),
    ("walk_istanbul_ferry", "istanbul ferry bosphorus seagulls"),
    ("walk_varanasi_ghats", "varanasi ghats morning walk boats"),
    ("gas_station", "gas station ambience pumps sounds"),
    ("car_repair", "car repair shop garage sounds impact wrench"),
    ("drive_through", "drive through order window sounds"),
    ("street_cleaning", "street sweeper cleaning truck sounds"),
    ("garbage_truck", "garbage truck collection morning sounds"),
    ("snowplow_street", "snowplow clearing street sounds"),
    ("bakery_morning", "bakery morning ambience oven trays"),
    ("coffee_roastery", "coffee roasting machine sounds"),
    ("brewery_tour", "brewery bottling line sounds tour"),
    ("fish_auction", "fish auction market morning shouting"),
    ("ferry_terminal", "ferry terminal announcements crowd"),
    ("subway_busker", "subway station busker music crowd passing"),
    ("botanic_garden", "botanical garden walk birds fountain"),
    ("waterpark_slides", "waterpark slides screaming splashing"),
    ("amusement_rides", "amusement park rides screams ambience"),
    ("ski_lift", "ski lift ride sounds wind"),
    ("sledding_hill", "sledding hill children winter sounds"),
    ("surf_beach", "surf beach waves surfers ambience"),
    ("dog_park", "dog park barking playing sounds"),
    ("horse_carriage", "horse carriage ride city clip clop"),
    ("rickshaw_ride", "cycle rickshaw ride street sounds"),
    ("rooftop_city", "rooftop view city traffic ambience below"),
    ("university_campus", "university campus walk between classes"),
    ("hospital_lobby", "hospital lobby waiting ambience"),
    ("hotel_lobby", "hotel lobby ambience fountain piano"),
    ("casino_floor", "casino floor slot machines ambience"),
    ("bingo_hall", "bingo hall caller ambience"),
    ("flea_market", "flea market browsing haggling sounds"),
    ("antique_shop", "antique shop clock ticking ambience"),
    ("pet_shop", "pet shop birds puppies sounds"),
    ("aviary_birds", "aviary tropical birds walk"),
    ("butterfly_house", "butterfly house greenhouse walk"),
    ("koi_pond", "koi pond garden fountain sounds"),
    ("windfarm_turbines", "wind turbines close sound field"),
    ("hydro_dam", "hydroelectric dam spillway sounds"),
    ("quarry_blast", "quarry machinery rock crusher sounds"),
    ("logging_forest", "logging chainsaw forest sounds"),
    ("orchard_picking", "apple orchard picking sounds autumn"),
    ("pumpkin_patch", "pumpkin patch farm visit sounds"),
    ("corn_maze", "corn maze walk rustling"),
    # wave 4 — distinct ambient SOUND sources, varied lengths
    ("icecream_truck", "ice cream truck jingle street children"),
    ("lawnmower_suburb", "lawn mowing suburb sounds neighborhood"),
    ("leafblower_fall", "leaf blower autumn leaves sounds"),
    ("pressure_washer", "pressure washing driveway sounds"),
    ("siren_passing", "ambulance siren passing street pedestrians"),
    ("crossing_bells", "railroad crossing bells cars waiting"),
    ("foghorn_harbor", "foghorn lighthouse fog harbor"),
    ("buoy_bell", "bell buoy waves sailing"),
    ("steamtrain_ride", "steam train ride whistle countryside"),
    ("balloon_burner", "hot air balloon burner ride"),
    ("seaplane_takeoff", "seaplane takeoff water sounds"),
    ("helicopter_tour", "helicopter tour cabin sounds city"),
    ("sail_flapping", "sailing boat sails flapping wind deck"),
    ("tent_rain", "rain on tent camping sounds"),
    ("creaky_house", "old house creaking floorboards tour"),
    ("cowbells_alps", "alps hiking cowbells meadow"),
    ("goat_herd", "goat herd bells crossing road"),
    ("call_to_prayer", "call to prayer street istanbul evening"),
    ("church_organ", "church organ playing visitors walking"),
    ("gospel_outside", "gospel choir heard from street"),
    ("drum_circle", "drum circle beach sunset crowd"),
    ("bagpipes_street", "bagpiper edinburgh street tourists"),
    ("accordion_metro", "accordion player metro passage"),
    ("moped_alley", "moped passing narrow alley italy"),
    ("elevator_ride", "old elevator ride sounds"),
    ("escalator_mall", "mall escalators ambience shoppers"),
    ("parking_garage", "parking garage echoes tires squeal"),
    ("toll_booth", "toll booth highway sounds"),
    ("car_ferry_load", "car ferry loading ramp sounds"),
    ("chairlift_summer", "summer chairlift ride mountain sounds"),
    ("zipline_forest", "zipline canopy tour sounds"),
    ("mtb_trail", "mountain bike trail pov sounds"),
    ("hooves_cobbles", "horse hooves cobblestone street"),
    ("fountain_show", "musical fountain show crowd"),
    ("sprinklers_park", "park sprinklers morning joggers"),
    ("hail_roof", "hail hitting roof car sounds"),
    ("creek_bridge", "creek under wooden bridge hiking"),
    ("tidepools_rocks", "tide pools rocky shore waves birds"),
    ("penguin_colony", "penguin colony sounds visitors"),
    ("sealions_dock", "sea lions barking pier tourists"),
    ("monkeys_temple", "monkeys temple stealing sounds tourists"),
    ("elephant_sanctuary", "elephant sanctuary sounds bathing"),
    ("starling_flock", "starling murmuration flock wings"),
    ("owls_night", "owl hooting night forest camera"),
    ("woodpecker_forest", "woodpecker drumming forest walk"),
    ("cicadas_summer", "cicadas loud summer walk"),
    ("thunder_porch", "thunderstorm from porch distant thunder"),
    ("typewriter_office", "typewriter museum office sounds"),
    ("clocktower_chimes", "clock tower chimes town square"),
    ("windchimes_porch", "wind chimes porch breeze"),
    ("schoolbell_recess", "school bell recess children running"),
    ("blackfriday_doors", "store opening rush crowd doors"),
    ("stadium_fireworks", "stadium fireworks celebration crowd"),
    ("newyear_street", "new years eve street celebration fireworks"),
    ("hanami_park", "cherry blossom park hanami crowd picnic"),
    ("oktoberfest_tent", "oktoberfest beer tent outside sounds"),
    ("christmas_market", "christmas market evening walk sounds"),
    ("ramadan_iftar", "ramadan iftar street cannon sounds"),
    ("chinese_newyear", "chinese new year firecrackers lion dance"),
    ("holi_festival", "holi festival colors crowd sounds"),
]


def source_yt(target: int = 70):
    STAGE.mkdir(parents=True, exist_ok=True)
    got = 0
    for name, query in YT_QUERIES:
        if got >= target:
            break
        tmp = STAGE / f"yt_{name}.mp4"
        tmp.unlink(missing_ok=True)
        sec = _clip_len()
        print(f"  [yt] {name} ({sec}s): {query}")
        r = subprocess.run(
            ["yt-dlp", f"ytsearch1:{query}",
             "--download-sections", f"*180-{180 + sec}",
             "-f", "mp4[height<=720]/best[height<=720]/best",
             "--no-playlist", "--force-keyframes-at-cuts",
             "--match-filter", "duration>240",
             "--download-archive", str(ARCHIVE),
             "--print-to-file", "%(webpage_url)s|%(title)s", str(STAGE / f"yt_{name}.meta"),
             "--no-warnings", "-o", str(tmp)],
            capture_output=True, text=True, timeout=420)
        if not tmp.exists():
            print(f"    skip ({(r.stderr or '')[:90].strip()})")
            continue
        meta_f = STAGE / f"yt_{name}.meta"
        url, title = ("", "")
        if meta_f.exists():
            parts = meta_f.read_text(encoding="utf-8").strip().split("|", 1)
            url = parts[0]
            title = parts[1] if len(parts) > 1 else ""
        if _accept(tmp, f"b3_{name}", {"source": "youtube", "url": url, "title": title,
                                       "license": "research use; not redistributed"}):
            got += 1
    print(f"[yt] accepted {got}")
    return got


# ---------------------------------------------------------------- source: Commons
WM_TERMS = ["walking tour city", "street market video", "harbor boats",
            "traffic street sound", "festival crowd", "rain street",
            "waterfall river", "construction site", "farm animals video",
            "airport", "playground", "beach waves people", "protest demonstration",
            "village street", "fountain square", "birds park city",
            # wave 2
            "parade procession", "bus station", "ferry port", "fair fairground",
            "carnival", "windmill", "helicopter", "fire truck", "storm wind",
            "river boat", "sheep goats", "school yard", "swimming", "snow ski",
            "night city", "temple ceremony", "church mass", "wedding dance"]


def _wm_api(params: dict) -> dict:
    qs = urllib.parse.urlencode({**params, "format": "json"})
    req = urllib.request.Request(
        f"https://commons.wikimedia.org/w/api.php?{qs}", headers=HDRS)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def source_wm(target: int = 45):
    STAGE.mkdir(parents=True, exist_ok=True)
    got = 0
    for term in WM_TERMS:
        if got >= target:
            break
        print(f"  [wm] search: {term}")
        try:
            data = _wm_api({"action": "query", "generator": "search",
                            "gsrsearch": f"{term} filetype:video", "gsrnamespace": "6",
                            "gsrlimit": "8", "prop": "videoinfo",
                            "viprop": "url|size|derivatives"})
        except Exception as e:
            print(f"    api error {type(e).__name__}")
            continue
        pages = (data.get("query") or {}).get("pages", {})
        per_term = 0
        for p in pages.values():
            if got >= target or per_term >= 3:
                break
            vi = (p.get("videoinfo") or [{}])[0]
            dur = float(vi.get("duration") or 0)
            if dur < 15:
                continue
            # smallest derivative >= 360p, else original if < 90 MB
            url, size = None, 0
            for d in sorted(vi.get("derivatives", []),
                            key=lambda d: int(d.get("height") or 0)):
                if int(d.get("height") or 0) >= 360:
                    url = d.get("src")
                    break
            if not url:
                if int(vi.get("size") or 0) > 90 * 1024 * 1024:
                    continue
                url = vi.get("url")
            title = p.get("title", "").replace("File:", "")
            slug = re.sub(r"[^a-z0-9]+", "_", title.lower())[:32].strip("_")
            tmp = STAGE / f"wm_{slug}.vid"
            try:
                req = urllib.request.Request(url, headers=HDRS)
                with urllib.request.urlopen(req, timeout=180) as r, open(tmp, "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                        if f.tell() > 120 * 1024 * 1024:
                            raise IOError("too big")
            except Exception as e:
                tmp.unlink(missing_ok=True)
                print(f"    dl fail {slug}: {type(e).__name__}")
                continue
            cut = STAGE / f"wm_{slug}.mp4"
            _cut_middle(tmp, cut)
            tmp.unlink(missing_ok=True)
            if cut.exists() and _accept(cut, f"b3_wm_{slug}",
                                        {"source": "wikimedia", "title": title,
                                         "url": f"https://commons.wikimedia.org/wiki/File:{urllib.parse.quote(title)}",
                                         "license": "CC (verify on Commons page)"}):
                got += 1
                per_term += 1
    print(f"[wm] accepted {got}")
    return got


# ---------------------------------------------------------------- source: IA
IA_QUERIES = [
    'travelogue color city', 'home movie color vacation sound',
    'street scenes color 16mm sound', 'documentary color village daily life',
    'promotional film color 1970s city', 'nature film color sound',
    # wave 2
    'tourism film color 1980s', 'industrial film color factory',
    'documentary color market bazaar', 'travel film color asia sound',
    'documentary color harbor fishing', 'news footage color street 1990s',
    # wave 3
    'educational film color transportation', 'documentary color festival celebration',
    'travel film color europe sound', 'documentary color river boat',
    'film color zoo animals sound', 'documentary color construction building',
]


def source_ia(target: int = 40):
    STAGE.mkdir(parents=True, exist_ok=True)
    got = 0
    for q in IA_QUERIES:
        if got >= target:
            break
        print(f"  [ia] search: {q}")
        qs = urllib.parse.urlencode({
            "q": f'{q} AND mediatype:(movies)', "fl[]": "identifier",
            "rows": "12", "output": "json"})
        try:
            req = urllib.request.Request(
                f"https://archive.org/advancedsearch.php?{qs}", headers=HDRS)
            with urllib.request.urlopen(req, timeout=30) as r:
                docs = json.loads(r.read())["response"]["docs"]
        except Exception as e:
            print(f"    search error {type(e).__name__}")
            continue
        per_q = 0
        for doc in docs:
            if got >= target or per_q >= 3:
                break
            ident = doc["identifier"]
            try:
                req = urllib.request.Request(
                    f"https://archive.org/metadata/{ident}", headers=HDRS)
                with urllib.request.urlopen(req, timeout=30) as r:
                    meta = json.loads(r.read())
                mp4s = [f for f in meta.get("files", [])
                        if f.get("name", "").endswith(".mp4")]
                if not mp4s:
                    continue
                f0 = mp4s[0]
                length = float(f0.get("length") or 0)
                if length < 120:
                    continue
                url = f"https://archive.org/download/{ident}/{urllib.parse.quote(f0['name'])}"
            except Exception:
                continue
            # two spaced scenes max per film
            for si, frac in enumerate((0.3, 0.7)):
                if got >= target or per_q >= 2:
                    break
                start = length * frac
                slug = re.sub(r"[^a-z0-9]+", "_", ident.lower())[:28].strip("_")
                cut = STAGE / f"ia_{slug}_{si}.mp4"
                subprocess.run(["ffmpeg", "-y", "-ss", f"{start:.0f}", "-i", url,
                                "-t", str(_clip_len()), "-c:v", "libx264", "-preset",
                                "veryfast", "-crf", "26", "-c:a", "aac", str(cut)],
                               capture_output=True, timeout=300)
                if cut.exists() and cut.stat().st_size > 500_000 and _accept(
                        cut, f"b3_ia_{slug}_{si}",
                        {"source": "internet_archive", "item": ident, "url": url,
                         "license": "public domain / verify item page"}):
                    got += 1
                    per_q += 1
                else:
                    cut.unlink(missing_ok=True)
    print(f"[ia] accepted {got}")
    return got


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    UNSORTED.mkdir(parents=True, exist_ok=True)
    total = 0
    if which in ("yt", "all"):
        total += source_yt()
    if which in ("wm", "all"):
        total += source_wm()
    if which in ("ia", "all"):
        total += source_ia()
    print(f"\nbatch3 total accepted: {total}")
