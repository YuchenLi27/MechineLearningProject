from datasets import load_dataset
import fasttext
import re
from collections import Counter
from sklearn.metrics import f1_score, confusion_matrix

DS = "Shelton1013/SwitchLingua_text"
TEXT_COL = "data_generation_result"
MODEL_PATH = "models/lid.176.bin"

# 过滤语言对（必要）
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
    n = len(langs)
    switch_next = [-100] * n
    dur3 = [-100] * n

    next_non_other = [-1] * n
    nxt = -1
    for i in range(n - 1, -1, -1):
        next_non_other[i] = nxt
        if langs[i] != "other":
            nxt = i

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

        j = next_non_other[t]
        if j == -1:
            switch_next[t] = -100
            continue

        next_lang = langs[j]
        sw = int(prev_lang[t] != next_lang)
        switch_next[t] = sw

        if sw == 1:
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

def is_target_pair(row):
    a = row["first_language"]
    b = row["second_language"]
    return set([a, b]) == set([LANG_A, LANG_B])

def main(max_samples=1000):
    ds = load_dataset(DS)
    split = "train" if "train" in ds else list(ds.keys())[0]
    data = ds[split]

    model = fasttext.load_model(MODEL_PATH)

    y_true_switch = []
    y_pred_switch_never = []

    y_true_dur = []
    # duration baseline 需要先统计多数类
    dur_counter = Counter()

    used = 0
    i = 0
    while used < max_samples and i < len(data):
        row = data[i]
        i += 1
        if not is_target_pair(row):
            continue

        text = row[TEXT_COL]
        toks = simple_tokenize(text)
        langs = [lid_token(t, model) for t in toks]
        sw, dur = make_switch_and_duration(langs)

        # collect switch labels (ignore -100)
        for v in sw:
            if v == -100:
                continue
            y_true_switch.append(v)
            y_pred_switch_never.append(0)  # baseline: never switch

        # collect duration labels only at true switch points (dur != -100)
        for d in dur:
            if d == -100:
                continue
            y_true_dur.append(d)
            dur_counter[d] += 1

        used += 1

    if not y_true_switch:
        print("No switch labels collected. Something is wrong with labeling or filtering.")
        return

    # Switch baseline
    f1_never = f1_score(y_true_switch, y_pred_switch_never, zero_division=0)
    switch_rate = sum(y_true_switch) / len(y_true_switch)

    # Duration baseline: always predict majority class among true switches
    if not y_true_dur:
        print("No duration labels collected (no switches found).")
        return
    maj_class = dur_counter.most_common(1)[0][0]
    y_pred_dur_maj = [maj_class] * len(y_true_dur)
    dur_acc = sum(int(a == b) for a, b in zip(y_true_dur, y_pred_dur_maj)) / len(y_true_dur)

    print("=== Baseline evaluation on", used, "samples of", LANG_A, "/", LANG_B, "===")
    print("Total switch label positions:", len(y_true_switch))
    print("True switch rate:", round(switch_rate, 4))
    print("Switch F1 (never-switch baseline):", round(f1_never, 4))
    print("Total true switch points (duration eval positions):", len(y_true_dur))
    print("Duration majority class:", maj_class, "counts:", dict(dur_counter))
    print("Duration accuracy (majority baseline):", round(dur_acc, 4))

    # Optional: confusion matrix for switch baseline
    cm = confusion_matrix(y_true_switch, y_pred_switch_never, labels=[0,1])
    print("Switch confusion matrix [[TN, FP],[FN, TP]]:")
    print(cm)

if __name__ == "__main__":
    main(max_samples=1000)
