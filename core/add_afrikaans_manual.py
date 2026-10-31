import json
import time
import requests

d = json.load(open("D:/Maker/Fakereal/projects/anima_ai/recursive_gpt/tokendict.json"))
session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0"})

# Common translations for remaining
manual = {
    'water': 'water',
    'wind': 'wind',
    'iron/metal': 'yster/metaal',
    'below/down': 'onder/af',
    'above/up': 'bo/op',
    'stop/end': 'stop/einde',
    'be/exist': 'wees/bestaan',
    'adjective/preposition/conjunction': 'byvoeglike naamwoord/voorsetsel/voegwoord',
    'I / self': 'ek / self',
    'other objects / thing': 'ander voorwerpe / ding',
    '0': 'nul',
    '1': 'een',
    'plus': 'plus',
    'question end / yes-no': 'vraag einde / ja-nee',
    'when / per': 'wanneer / per',
    'prefix (noun-forming)': 'voorvoegsel (selfstandige naamwoord-vormend)',
    'because': 'omdat',
    'if/unless': 'as/tenzij',
    'this/that': 'hierdie/dat',
    'who/what': 'wie/wat',
    'root compound marker': 'wortel saamgestelde merker',
    'human body': 'menslike liggaam',
    'now/time present': 'nou/tyd teenwoordig',
    'action verb': 'aksiewerkwoord',
    'present tense': 'teenwoordige tyd',
    'past tense': 'verlede tyd',
    'future tense': 'toekomende tyd',
}

updated = 0
for cat, items in d.items():
    if cat.startswith("_"): continue
    if isinstance(items, dict):
        for word, val in items.items():
            if isinstance(val, list) and len(val) == 2:
                en = str(val[1])
                af = manual.get(en) or manual.get(en.lower())
                if not af:
                    try:
                        resp = session.get(f"https://api.mymemory.translated.net/get", params={"q": en, "langpair": "en|af"}, timeout=10)
                        data = resp.json()
                        af = data.get("responseData", {}).get("translatedText", "")
                        time.sleep(0.3)
                    except:
                        af = ""
                if af and af.strip():
                    val.append(af.strip())
                    updated += 1

json.dump(d, open("D:/Maker/Fakereal/projects/anima_ai/recursive_gpt/tokendict.json","w"), ensure_ascii=False, indent=2)
print("done", updated)
