import tkinter as tk
from tkinter import ttk
import json
from pathlib import Path

class App:
    def __init__(self, root):
        self.root = root
        root.title("Draekonic Translator - EN/AF <-> Draekonic")
        root.geometry("650x450")
        root.resizable(False, False)
        f1 = ttk.LabelFrame(root, text="Input", padding=10)
        f1.pack(fill="x", padx=10, pady=5)
        self.inp = tk.Text(f1, height=5, width=75)
        self.inp.pack()
        lf = ttk.Frame(root)
        lf.pack(fill="x", padx=10, pady=5)
        ttk.Label(lf, text="From:").pack(side="left")
        self.fr = ttk.Combobox(lf, values=["English", "Afrikaans", "Draekonic"], width=12)
        self.fr.set("English")
        self.fr.pack(side="left", padx=2)
        ttk.Button(lf, text="Translate", command=self.go).pack(side="left", padx=10)
        ttk.Label(lf, text="To:").pack(side="left")
        self.to = ttk.Combobox(lf, values=["Draekonic", "English", "Afrikaans"], width=12)
        self.to.set("Draekonic")
        self.to.pack(side="left", padx=2)
        ttk.Button(lf, text="Swap", command=self.swap).pack(side="left", padx=5)
        f2 = ttk.LabelFrame(root, text="Output", padding=10)
        f2.pack(fill="x", padx=10, pady=5)
        self.out = tk.Text(f2, height=5, width=75)
        self.out.pack()
        self.st = ttk.Label(root, text="Ready")
        self.st.pack(pady=5)
        self.load()
    
    def load(self):
        dict_path = Path(r"D:/Maker/Fakereal/projects/anima_ai/typus_animus/tokendict.json")
        with open(dict_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        self.en2 = {}
        self.af2 = {}
        self.d2en = {}
        self.d2af = {}
        for c, it in d.items():
            if c.startswith("_"):
                continue
            if isinstance(it, dict):
                for w, v in it.items():
                    if isinstance(v, (list, tuple)) and len(v) >= 2:
                        en = str(v[1]).lower().strip()
                        if en:
                            for p in en.replace("/", ",").split(","):
                                p = p.strip()
                                if p and p not in self.en2:
                                    self.en2[p] = w
                            self.d2en[w] = v[1]
                        if len(v) >= 3:
                            af = str(v[2]).lower().strip()
                            if af:
                                for p in af.replace("/", ",").split(","):
                                    p = p.strip()
                                    if p and p not in self.af2:
                                        self.af2[p] = w
                                self.d2af[w] = v[2]
        self.st.config(text=str(len(self.en2)) + " EN + " + str(len(self.af2)) + " AF entries")
    
    def go(self):
        t = self.inp.get("1.0", tk.END).strip()
        if not t:
            return
        fr, to = self.fr.get(), self.to.get()
        self.out.delete("1.0", tk.END)
        if fr == "Draekonic" and to in ["English", "Afrikaans"]:
            r = []
            for w in t.split():
                if w in self.d2en and to == "English":
                    r.append(self.d2en[w])
                elif w in self.d2af and to == "Afrikaans":
                    r.append(self.d2af[w])
                elif w in self.d2en:
                    r.append(self.d2en[w])
                else:
                    r.append(w)
            self.out.insert("1.0", " / ".join(r))
        elif to == "Draekonic":
            ws = t.lower().replace(".", " ").replace(",", " ").replace("!", " ").replace("?", " ").split()
            r = []
            for w in ws:
                if fr == "English" and w in self.en2:
                    r.append(self.en2[w])
                elif fr == "Afrikaans" and w in self.af2:
                    r.append(self.af2[w])
                elif w in self.en2:
                    r.append(self.en2[w])
                elif w in self.af2:
                    r.append(self.af2[w])
                else:
                    r.append("[" + w + "]")
            self.out.insert("1.0", " ".join(r))
        else:
            self.out.insert("1.0", t)
    
    def swap(self):
        fr, to = self.fr.get(), self.to.get()
        self.fr.set(to)
        self.to.set(fr)
        i = self.inp.get("1.0", tk.END)
        o = self.out.get("1.0", tk.END)
        self.inp.delete("1.0", tk.END)
        self.out.delete("1.0", tk.END)
        self.inp.insert("1.0", o.strip())
        self.out.insert("1.0", i.strip())


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
