import json
import re
import ast
import unicodedata
from datasets import load_dataset
from tqdm import tqdm

DS = "Shelton1013/SwitchLingua_text"
OUT = "data_processed/weak_cs.jsonl"

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

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

def parse_turns(value):
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

def main(limit=20000, max_tokens=256):
    ds = load_dataset(DS)["train"]

    # ensure output directory exists
    import os
    os.makedirs("data_processed", exist_ok=True)

    written = 0
    with open(OUT, "w", encoding="utf-8") as f:
        for ex in tqdm(ds, total=len(ds)):
            turns = parse_turns(ex["data_generation_result"])
            text = " ".join(turns)

            # keep only Arabic+Latin mixed (robust)
            if not (has_arabic_by_range(text) and has_latin_by_range(text)):
                continue

            tokens = tokenize(text)[:max_tokens]
            tags = weak_lang_tags(tokens)
            switch_next, dur3 = build_labels(tags)

            rec = {
                "first_language": ex["first_language"],
                "second_language": ex["second_language"],
                "tokens": tokens,
                "lang_tags": tags,
                "switch_next": switch_next,
                "dur3": dur3,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            if written >= limit:
                break

    print(f"Wrote {written} examples to {OUT}")

if __name__ == "__main__":
    main()
