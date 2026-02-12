from torch.utils.data import Dataset
from datasets import load_dataset
import fasttext
import re

DS = "Shelton1013/SwitchLingua_text"
TEXT_COL = "data_generation_result"
MODEL_PATH = "models/lid.176.bin"

LANG_A = "English"
LANG_B = "Chinese"

def simple_tokenize(text: str):
    pattern = r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[\u4e00-\u9fff]|[^\s]"
    return re.findall(pattern, text)

def lid_token(tok: str, model) -> str:
    t = tok.strip()
    if t == "":
        return "other"
    if len(t) == 1 and ("\u4e00" <= t <= "\u9fff"):
        return "zh"
    if all(not c.isalnum() for c in t):
        return "other"
    if t.isascii() and any(c.isalpha() for c in t) and all((c.isalpha() or c == "'") for c in t):
        return "en"
    labels, probs = model.predict(t)
    lang = labels[0].replace("__label__", "")
    if lang.startswith("en"):
        return "en"
    if lang.startswith("zh"):
        return "zh"
    return "other"

def make_switch_and_duration(langs, small_max=2, med_max=5):
    """
    Skip-other label:
    - current_lang: 最近的非 other 语言（在 t 或 t 左侧）
    - next_lang: t 右侧第一个非 other 语言
    - switch_next[t] = 1 if current_lang != next_lang
    - duration3[t]：只在 switch 点定义，统计 next_lang 语言段长度（忽略 other）
    """
    n = len(langs)
    switch_next = [-100] * n
    dur3 = [-100] * n

    # next non-other index for each t
    next_non_other = [-1] * n
    nxt = -1
    for i in range(n - 1, -1, -1):
        next_non_other[i] = nxt
        if langs[i] != "other":
            nxt = i

    # prev non-other language for each t
    prev_lang = ["other"] * n
    last = "other"
    for i in range(n):
        if langs[i] != "other":
            last = langs[i]
        prev_lang[i] = last

    for t in range(n):
        cur = prev_lang[t]
        j = next_non_other[t]
        if cur == "other" or j == -1:
            switch_next[t] = -100
            continue

        nxt_lang = langs[j]
        sw = int(cur != nxt_lang)
        switch_next[t] = sw

        if sw == 1:
            # count length of upcoming segment starting at j (ignore other)
            L = 0
            k = j
            while k < n:
                if langs[k] == "other":
                    k += 1
                    continue
                if langs[k] != nxt_lang:
                    break
                L += 1
                k += 1

            if L <= small_max:
                dur3[t] = 0
            elif L <= med_max:
                dur3[t] = 1
            else:
                dur3[t] = 2

    return switch_next, dur3


def is_target_pair(row):
    return set([row["first_language"], row["second_language"]]) == set([LANG_A, LANG_B])

class SwitchLinguaCSDataset(Dataset):
    def __init__(self, split="train", max_samples=5000, seed=0):
        self.ds = load_dataset(DS)[split]
        self.model = fasttext.load_model(MODEL_PATH)

        # 先筛出语言对的索引
        idxs = []
        for i in range(len(self.ds)):
            if is_target_pair(self.ds[i]):
                idxs.append(i)
                if len(idxs) >= max_samples:
                    break

        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, k):
        row = self.ds[self.idxs[k]]
        text = row[TEXT_COL]

        toks = simple_tokenize(text)
        langs = [lid_token(t, self.model) for t in toks]
        sw, dur = make_switch_and_duration(langs)

        return {
            "tokens": toks,
            "switch_labels": sw,
            "dur_labels": dur,
        }
