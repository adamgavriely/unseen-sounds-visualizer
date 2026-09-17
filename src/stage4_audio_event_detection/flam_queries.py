"""The FLAM query vocabulary (docs/prereg_v4.md §4). Fixed before the run; not edited after.

FLAM was trained on descriptive captions, so each query is a short phrase, and each maps
to ONE AudioSet label so that everything downstream of the detector -- src/labels
families, the gate, dedup, depiction -- is unchanged. Written from the ontology and the
pipeline's families; the five entries marked "dev" were added so that every real labelled
detection on the dev split (benchmark/dev_phantoms.json) has a query. Labels the pipeline never shows (speech-like, scene-level) are left out.
"""

QUERIES = [
    # AudioSet label                          descriptive query
    # -- road / rail / air / water transport
    ("Vehicle horn, car horn, honking",       "a car horn honking"),
    ("Car",                                   "a car engine driving past"),
    ("Truck",                                 "a heavy truck engine rumbling"),
    ("Bus",                                   "a bus engine idling and pulling away"),
    ("Motorcycle",                            "a motorcycle engine revving"),
    ("Reversing beeps",                       "a reversing vehicle beeping"),
    ("Skidding",                              "tyres skidding and screeching"),
    ("Air horn, truck horn",                  "an air horn blasting"),
    ("Train",                                 "a train passing on the rails"),
    ("Train horn",                            "a train horn blaring"),
    ("Subway, metro, underground",            "a subway train arriving in a station"),
    ("Helicopter",                            "a helicopter rotor thumping overhead"),
    ("Fixed-wing aircraft, airplane",         "an airplane jet flying overhead"),
    ("Motorboat, speedboat",                  "a motorboat engine on the water"),
    ("Bicycle bell",                          "a bicycle bell ringing"),
    ("Skateboard",                            "a skateboard rolling on pavement"),
    # -- sirens, alarms, bells, phones
    ("Siren",                                 "an emergency vehicle siren wailing"),
    ("Civil defense siren",                   "an air raid siren rising and falling"),
    ("Fire alarm",                            "a fire alarm bell ringing"),
    ("Smoke detector, smoke alarm",           "a smoke detector beeping"),
    ("Car alarm",                             "a car alarm going off"),
    ("Alarm clock",                           "an alarm clock ringing"),
    ("Buzzer",                                "a buzzer buzzing"),
    ("Beep, bleep",                           "an electronic beep"),
    ("Telephone bell ringing",                "a telephone ringing"),
    ("Doorbell",                              "a doorbell ringing"),
    ("Church bell",                           "a church bell tolling"),
    ("Bell",                                  "a bell ringing"),
    ("Wind chime",                            "wind chimes tinkling"),
    ("Whistle",                               "a whistle blowing"),
    # -- animals
    ("Dog",                                   "a dog barking"),
    ("Cat",                                   "a cat meowing"),
    ("Bird",                                  "birds chirping and singing"),
    ("Crow",                                  "a crow cawing"),
    ("Owl",                                   "an owl hooting"),
    ("Chicken, rooster",                      "a rooster crowing"),
    ("Duck",                                  "ducks quacking"),
    ("Goose",                                 "geese honking"),                       # dev
    ("Honk",                                  "a goose honking loudly"),              # dev
    ("Horse",                                 "a horse neighing and hooves clip-clopping"),
    ("Sheep",                                 "a sheep bleating"),
    ("Cattle, bovinae",                       "a cow mooing"),
    ("Pig",                                   "a pig oinking"),
    ("Insect",                                "insects buzzing"),
    ("Frog",                                  "frogs croaking"),
    ("Roaring cats (lions, tigers)",          "a lion roaring"),
    # -- people, non-speech
    ("Laughter",                              "people laughing"),
    ("Crying, sobbing",                       "a person crying and sobbing"),
    ("Baby cry, infant cry",                  "a baby crying"),
    ("Cough",                                 "a person coughing"),
    ("Sneeze",                                "a person sneezing"),
    ("Applause",                              "an audience applauding"),
    ("Clapping",                              "hands clapping"),
    ("Cheering",                              "a crowd cheering"),
    ("Crowd",                                 "a crowd of people talking in the background"),
    ("Walk, footsteps",                       "footsteps walking on the floor"),
    ("Knock",                                 "knocking on a door"),
    ("Door",                                  "a door opening and closing"),
    ("Slam",                                  "a door slamming shut"),
    ("Sliding door",                          "a sliding door rolling open"),          # dev
    ("Typing",                                "typing on a keyboard"),
    ("Zipper (clothing)",                     "a zipper being zipped"),                # dev
    # -- household
    ("Water tap, faucet",                     "water running from a tap"),
    ("Dishes, pots, and pans",                "dishes and pots clattering"),
    ("Frying (food)",                         "food frying and sizzling in a pan"),
    ("Blender",                               "a blender whirring"),
    ("Vacuum cleaner",                        "a vacuum cleaner running"),
    ("Toilet flush",                          "a toilet flushing"),
    ("Microwave oven",                        "a microwave oven humming and beeping"),
    ("Shatter",                               "glass shattering and breaking"),
    # -- tools, machines
    ("Chainsaw",                              "a chainsaw cutting wood"),
    ("Drill",                                 "a power drill drilling"),
    ("Hammer",                                "a hammer hitting nails"),
    ("Lawn mower",                            "a lawn mower engine running"),
    ("Jackhammer",                            "a jackhammer pounding the pavement"),
    # -- nature, weather, fire
    ("Rain",                                  "rain falling"),
    ("Thunder",                               "thunder rumbling"),
    ("Waves, surf",                           "ocean waves crashing on the shore"),
    ("Stream",                                "a stream of water flowing"),
    ("Fire",                                  "a fire crackling"),
    ("Crack",                                 "a sharp cracking snap"),                # dev
    # -- bangs
    ("Gunshot, gunfire",                      "gunshots firing"),
    ("Explosion",                             "an explosion booming"),
    ("Fireworks",                             "fireworks exploding in the sky"),
]

LABELS = [l for l, _ in QUERIES]
TEXTS = [t for _, t in QUERIES]
assert len(set(LABELS)) == len(LABELS), "one query per label"
