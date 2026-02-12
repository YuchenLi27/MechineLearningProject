# src/pipeline_preview.py
from datasets import load_dataset
import fasttext
import re

DS = "Shelton1013/SwitchLingua_text"
TEXT_COL = "data_generation_result"
MODEL_PATH = "models/lid.176.bin"

# 你要筛选的语言对（先按 HF 字段里常见写法；若不匹配，改这里）
LANG_A = "English"
LANG_B = "Chinese"

def simple_tokenize(text: str):
    """
    中英混合情况下够用的 tokenizer：
    - 英文按词/数字/标点拆
    - 中文按单字
    - 其它字符按单字符/符号保留
    """
    # 英文词、数字、单个中文、其它非空白单字符
    pattern = r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[\u4e00-\u9fff]|[^\s]"
    return re.findall(pattern, text)

def lid_token(tok: str, model) -> str:
    """
    将 token 归为 en / zh / other
    - 规则优先：中文单字->zh；纯英文串->en；标点->other
    - 其余交给 fastText；只接受 en/zh，其它->other
    """
    t = tok.strip()
    if t == "":
        return "other"

    # 中文（单个 CJK 字）
    if len(t) == 1 and ("\u4e00" <= t <= "\u9fff"):
        return "zh"

    # 标点/符号
    if all(not c.isalnum() for c in t):
        return "other"

    # 纯英文 token（避免 fastText 对短词不稳）
    if t.isascii() and any(c.isalpha() for c in t) and all((c.isalpha() or c == "'") for c in t):
        return "en"

    labels, probs = model.predict(t)
    lang = labels[0].replace("__label__", "")  # e.g., "en", "zh", "ar", ...
    if lang.startswith("en"):
        return "en"
    if lang.startswith("zh"):
        return "zh"
    return "other"

def make_switch_and_duration(langs, small_max=2, med_max=5):
    """
    改进版：
    - switch_next[t]：比较“当前位置 t 的最近非 other 语言”和“t 后面第一个非 other 语言”
      如果不同，则认为在 t 处预测到了下一次切换。
    - duration3[t]：只在 switch 点定义：从“下一个非 other token”开始的新语言段长度（只数非 other）
    """
    n = len(langs)
    switch_next = [-100] * n
    dur3 = [-100] * n

    # 先预处理：每个位置 i 右侧第一个非 other 的位置 next_non_other[i]
    next_non_other = [-1] * n
    nxt = -1
    for i in range(n - 1, -1, -1):
        next_non_other[i] = nxt
        if langs[i] != "other":
            nxt = i

    # 再预处理：每个位置 i 左侧最近非 other 的语言 prev_lang[i]
    prev_lang = ["other"] * n
    last = "other"
    for i in range(n):
        if langs[i] != "other":
            last = langs[i]
        prev_lang[i] = last

    for t in range(n):
        if prev_lang[t] == "other":
            switch_next[t] = 0
            continue

        j = next_non_other[t]  # t 右侧第一个非 other token
        if j == -1:
            switch_next[t] = -100
            continue

        next_lang = langs[j]
        sw = int(prev_lang[t] != next_lang)
        switch_next[t] = sw

        if sw == 1:
            # 从 j 开始数新语言段长度（只数非 other）
            L = 0
            k = j
            while k < n:
                if langs[k] == "other":
                    k += 1
                    continue
                if langs[k] != next_lang:
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


def find_language_pair_indices(data, lang_a, lang_b, limit=5):
    """
    在数据集中找 first_language/second_language 等于指定语言对（无序）的行索引
    """
    idxs = []
    target = set([lang_a, lang_b])

    for i in range(len(data)):
        a = data[i].get("first_language", None)
        b = data[i].get("second_language", None)
        if a is None or b is None:
            continue
        if set([a, b]) == target:
            idxs.append(i)
            if len(idxs) >= limit:
                break
    return idxs

def main():
    ds = load_dataset(DS)
    split = list(ds.keys())[0]
    data = ds[split]

    print("split:", split, "rows:", len(data))
    print("target language pair:", LANG_A, "/", LANG_B)

    model = fasttext.load_model(MODEL_PATH)

    idxs = find_language_pair_indices(data, LANG_A, LANG_B, limit=5)
    print("found samples:", len(idxs))
    if not idxs:
        print("No samples found for that pair.")
        print("Run a quick language distribution script or print a few rows' first_language/second_language to see exact names.")
        return

    # 只预览第一条
    idx = idxs[0]
    row = data[idx]
    text = row[TEXT_COL]

    toks = simple_tokenize(text)
    langs = [lid_token(t, model) for t in toks]
    sw, dur = make_switch_and_duration(langs)

    print("=" * 80)
    print("row idx:", idx)
    print("first_language:", row["first_language"], "second_language:", row["second_language"])
    print("text preview:", text[:300].replace("\n", " ") + ("..." if len(text) > 300 else ""))
    print("=" * 80)

    # 打印前 150 个 token 对齐情况
    for t, l, s, d in list(zip(toks, langs, sw, dur))[:150]:
        print(f"{t:>12}\t{l}\tSW={s}\tDUR={d}")

    print("=" * 80)
    hits = [(i, toks[i], langs[i], sw[i], dur[i]) for i in range(min(200, len(sw))) if sw[i] == 1]
    print("SW=1 hits (first 200 positions):", len(hits))
    for h in hits[:10]:
        print("pos:", h[0], "token:", h[1], "lang_at_token:", h[2], "DUR:", h[4])


if __name__ == "__main__":
    main()
