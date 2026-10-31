import json
import time
import urllib.request
import urllib.parse

d = json.load(open("D:/Maker/Fakereal/projects/anima_ai/recursive_gpt/tokendict.json"))
def translate(text):
    if not text or text.strip() == "":
        return ""
    try:
        q = urllib.parse.quote(text)
        url = f"https://api.mymemory.translated.net/get?q={q}&langpair=en|af"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        return data.get("responseData", {}).get("translatedText", "")
    except Exception as e:
        return ""

updated = 0
for cat, items in d.items():
    if cat.startswith("_"): continue
    if isinstance(items, dict):
        for word, val in items.items():
            if isinstance(val, (list, tuple)) and len(val) >= 2:
                en = str(val[1])
                # check if already has afrikaans (has / or contains afrikaans chars pattern)
                if "/" in en or len(val) >= 3:
                    continue
                af = translate(en)
                if af and af.lower() != en.lower():
                    val.append(af)
                    updated += 1
                    if updated % 10 == 0:
                        print(f"updated {updated}")
                        json.dump(d, open("D:/Maker/Fakereal/projects/anima_ai/recursive_gpt/tokendict.json","w"), ensure_ascii=False, indent=2)
                    time.sleep(0.5)
            elif isinstance(val, list) and len(val) == 1:
                # try to add empty placeholder? skip
                pass

json.dump(d, open("D:/Maker/Fakereal/projects/anima_ai/recursive_gpt/tokendict.json","w"), ensure_ascii=False, indent=2)
print("done", updated)
