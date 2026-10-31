#!/usr/bin/env python3
"""English/Afrikaans to Draekonic translator using tokendict.json"""
import json
import sys
import os
from pathlib import Path

class DraekTranslator:
    def __init__(self, dict_path=None):
        if dict_path is None:
            dict_path = Path(__file__).parent / "tokendict.json"
        with open(dict_path, 'r', encoding='utf-8') as f:
            self.tokendict = json.load(f)
        self.build_maps()
        
    def build_maps(self):
        self.en_to_draek = {}
        self.af_to_draek = {}
        self.draek_to_en = {}
        self.draek_to_af = {}
        self.entries = {}
        
        for cat, items in self.tokendict.items():
            if cat.startswith('_'): continue
            if isinstance(items, dict):
                for word, val in items.items():
                    if isinstance(val, (list, tuple)) and len(val) >= 2:
                        self.entries[word] = val
                        en = str(val[1]).lower().strip()
                        if en:
                            # handle multiple meanings separated by /
                            for part in en.replace('/', ',').split(','):
                                part = part.strip()
                                if part and part not in self.en_to_draek:
                                    self.en_to_draek[part] = word
                            self.draek_to_en[word] = val[1]
                        if len(val) >= 3:
                            af = str(val[2]).lower().strip()
                            if af:
                                for part in af.replace('/', ',').split(','):
                                    part = part.strip()
                                    if part and part not in self.af_to_draek:
                                        self.af_to_draek[part] = word
                                self.draek_to_af[word] = val[2]
    
    def translate_en(self, text):
        words = text.lower().replace('.', ' ').replace(',', ' ').replace('!', ' ').replace('?', ' ').split()
        result = []
        for w in words:
            if w in self.en_to_draek:
                result.append(self.en_to_draek[w])
            elif w in self.af_to_draek:  # try af
                result.append(self.af_to_draek[w])
            else:
                result.append('[%s]' % w)
        return ' '.join(result)
    
    def translate_af(self, text):
        words = text.lower().replace('.', ' ').replace(',', ' ').replace('!', ' ').replace('?', ' ').split()
        result = []
        for w in words:
            if w in self.af_to_draek:
                result.append(self.af_to_draek[w])
            elif w in self.en_to_draek:
                result.append(self.en_to_draek[w])
            else:
                result.append('[%s]' % w)
        return ' '.join(result)
    
    def translate(self, text, lang='en'):
        if lang.lower() in ['af', 'afrikaans']:
            return self.translate_af(text)
        return self.translate_en(text)
    
    def reverse(self, draek_text):
        words = draek_text.split()
        result = []
        for w in words:
            if w in self.draek_to_en:
                result.append(self.draek_to_en[w])
            elif w in self.draek_to_af:
                result.append(self.draek_to_af[w])
            else:
                result.append(w)
        return ' / '.join(result)

def main():
    t = DraekTranslator()
    if len(sys.argv) < 2:
        print("Usage: draek_translator.py <text> [en|af]")
        print("Or: draek_translator.py --reverse <draekonic>")
        sys.exit(1)
    if sys.argv[1] == '--reverse':
        print(t.reverse(' '.join(sys.argv[2:])))
        sys.exit(0)
    lang = sys.argv[2] if len(sys.argv) > 2 else 'en'
    text = ' '.join(sys.argv[1:]) if len(sys.argv) == 2 else ' '.join(sys.argv[1:-1]) if lang in ['en','af'] else ' '.join(sys.argv[1:])
    if len(sys.argv) > 2 and sys.argv[2] in ['en','af']:
        text = ' '.join(sys.argv[1:-1])
    print(t.translate(text, lang))

if __name__ == '__main__':
    main()
