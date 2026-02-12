import re
import random
import unicodedata
import ast
from collections import Counter
from datasets import load_dataset

DS = "Shelton1013/SwitchLingua_text"

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")
LATIN_RE = re.compile(r"[A-Za-z]")

def tokenize(text: str):
    return _TOKEN_RE.findall(text)

def has_arabic_by_range(text: str) -> bool:
    for ch in text:
        o = ord(ch)
        if (0x0600 <= o <= 0x06FF) or (0x0750 <= o <= 0x077F) or (0x08A0 <= o <= 0x08FF) or (0xFB50 <= o <= 0xFDFF) or (0xFE70 <= o <= 0xFEFF):
            return True
    return False

def has_latin_by_range(text: str) -> bool:
    for ch in text:
        o = ord(ch)
        if (0x41 <= o <= 0x5A) or (0x61 <= o <= 0x7A):
            return True
        if (0x00C0 <= o <= 0x024F) or (0x1E00 <= o <= 0x1EFF):
            return True
    return False

def char_script(ch: str) -> str:
    o = ord(ch)
    if (0x0600 <= o <= 0x06FF) or (0x0750 <= o <= 0x077F) or (0x08A0 <= o <= 0x08FF) or (0xFB50 <= o <= 0xFDFF) or (0xFE70 <= o <= 0xFEFF):
        return "ARABIC"
    if (0x0041 <= o <= 0x007A) or (0x00C0 <= o <= 0x024F) or (0x1E00 <= o <= 0x1EFF):
        return "LATIN"
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return "OTHER"
    if "ARABIC" in name:
        return "ARABIC"
    if "LATIN" in name:
        return "LATIN"
    return "OTHER"

def token_script(tok: str) -> str:
    for ch in tok:
        if ch.isspace():
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("P"):
            continue
        sc = char_script(ch)
        if sc != "OTHER":
            return sc
    return "OTHER"

def weak_lang_tags(tokens):
    raw = []
    for tok in tokens:
        sc = token_script(tok)
        if sc == "ARABIC":
            raw.append("L1")
        elif sc == "LATIN":
            raw.append("L2")
        else:
            raw.append("O")

    tags = []
    last = "L1"
    for t in raw:
        if t == "O":
            tags.append(last)
        else:
            tags.append(t)
            last = t
    return tags

def build_labels(tags):
    n = len(tags)
    switch_next = [-100] * n
    dur3 = [-100] * n
    for t in range(n - 1):
        sw = int(tags[t] != tags[t + 1])
        switch_next[t] = sw
        if sw == 1:
            L = 1
            while (t + 1 + L) < n and tags[t + 1 + L] == tags[t + 1]:
                L += 1
            dur3[t] = 0 if L <= 2 else (1 if L <= 5 else 2)
    return switch_next, dur3

def parse_turns(value):
    """
    Normalize data_generation_result into list[str].
    - if already list: use it
    - if str that looks like "['a','b']": literal_eval into list
    - otherwise: treat as one-turn list
    """
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except Exception:
                pass
        return [s]
    return [str(value)]

def main():
    ds = load_dataset(DS)["train"]

    # -------- Row0 sanity check --------
    ex0 = ds[0]
    turns0 = parse_turns(ex0["data_generation_result"])
    text0 = " ".join(turns0)

    has_ar0_rng = has_arabic_by_range(text0)
    has_la0_rng = has_latin_by_range(text0)
    has_ar0_re = bool(ARABIC_RE.search(text0))
    has_la0_re = bool(LATIN_RE.search(text0))

    toks0 = tokenize(text0)
    tags0 = weak_lang_tags(toks0)
    has_L1_tok = any(t == "L1" for t in tags0)
    has_L2_tok = any(t == "L2" for t in tags0)

    print("DEBUG row0 original type:", type(ex0["data_generation_result"]))
    print("DEBUG row0 parsed turns len:", len(turns0))
    print("DEBUG row0 repr head:", repr(text0[:200]))
    print("DEBUG row0 range has_ar:", has_ar0_rng, "has_latin:", has_la0_rng)
    print("DEBUG row0 regex  has_ar:", has_ar0_re, "has_latin:", has_la0_re)
    print("DEBUG row0 tags   has_L1:", has_L1_tok, "has_L2:", has_L2_tok)

    # -------- Count kept in first 500 (range Arabic+Latin) --------
    checked = 500
    kept = 0
    kept_indices = []

    for i in range(min(len(ds), checked)):
        ex = ds[i]
        turns = parse_turns(ex["data_generation_result"])
        text = " ".join(turns)

        if i == 0:
            print("\nDEBUG LOOP i=0 original type:", type(ex["data_generation_result"]))
            print("DEBUG LOOP i=0 parsed turns len:", len(turns))
            print("DEBUG LOOP i=0 repr head:", repr(text[:200]))
            print("DEBUG LOOP i=0 range ar/latin:", has_arabic_by_range(text), has_latin_by_range(text))

        if has_arabic_by_range(text) and has_latin_by_range(text):
            kept += 1
            kept_indices.append(i)

    print(f"\nTotal rows: {len(ds)}")
    print(f"Kept in first {checked} (range Arabic+Latin): {kept}")

    if kept == 0:
        print("Still 0 kept; then either parsing failed or these first 500 have no mix (unlikely).")
        return

    # -------- Show one sample + labels --------
    idx = random.choice(kept_indices)
    ex = ds[idx]
    turns = parse_turns(ex["data_generation_result"])
    text = " ".join(turns)

    toks = tokenize(text)[:160]
    tags = weak_lang_tags(toks)
    sw, dur = build_labels(tags)

    print("\n--- SAMPLE ---")
    print("row:", idx, "first_language:", ex["first_language"], "second_language:", ex["second_language"])
    print("range filter ar/latin:", has_arabic_by_range(text), has_latin_by_range(text))
    print("tag counts:", dict(Counter(tags)))

    for i, (tok, tg) in enumerate(zip(toks, tags)):
        mark = "SW" if sw[i] == 1 else "  "
        d = dur[i] if dur[i] != -100 else ""
        print(f"{i:03d} {tok:>14}  {tg}  {mark}  {d}")

if __name__ == "__main__":
    main()
