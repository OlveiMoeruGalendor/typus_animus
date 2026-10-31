import json
import time
import requests

d = json.load(open("D:/Maker/Fakereal/projects/anima_ai/recursive_gpt/tokendict.json"))
session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0"})

updated = 0
to_update = []
for cat, items in d.items():
    if cat.startswith("_"): continue
    if isinstance(items, dict):
        for word, val in items.items():
            if isinstance(val, list) and len(val) == 2:
                en = str(val[1])
                if "/" not in en:
                    to_update.append((cat, word, val, en))

print(f"Need to update {len(to_update)} entries")

for i, (cat, word, val, en) in enumerate(to_update):
    try:
        resp = session.get(f"https://api.mymemory.translated.net/get", params={"q": en, "langpair": "en|af"}, timeout=10)
        data = resp.json()
        af = data.get("responseData", {}).get("translatedText", "")
        if af and af.strip() and af.lower().strip() != en.lower().strip():
            val.append(af.strip())
            updated += 1
            if updated % 20 == 0:
                print(f"updated {updated}/{len(to_update)}")
                json.dump(d, open("D:/Maker/Fakereal/projects/anima_ai/recursive_gpt/tokendict.json","w"), ensure_ascii=False, indent=2)
        time.sleep(0.3)
    except Exception as e:
        pass

json.dump(d, open("D:/Maker/Fakereal/projects/anima_ai/recursive_gpt/tokendict.json","w"), ensure_ascii=False, indent=2)
print("done", updated)
