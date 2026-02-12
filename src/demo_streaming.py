# src/demo_streaming.py
import torch
import re
from transformers import XLMRobertaTokenizerFast

from src.model_mt import XLMRMultiTask

CKPT_PATH = "runs/xlmr_streaming.pt"
MODEL_NAME = "xlm-roberta-base"

# 来自你当前 eval 的最佳阈值
SWITCH_THR_ON = 0.45       # 触发阈值
SWITCH_THR_OFF = 0.35      # 解除阈值（hysteresis，避免抖动）

MAX_LEN = 192
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 触发后冷却 N 个 token（避免标点连触发）
COOLDOWN_STEPS = 6

def simple_tokenize(text: str):
    pattern = r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[\u4e00-\u9fff]|[^\s]"
    return re.findall(pattern, text)

def is_punct(tok: str) -> bool:
    t = tok.strip()
    return t != "" and all(not c.isalnum() for c in t) and not ("\u4e00" <= t <= "\u9fff")

def dur_label_to_str(d: int) -> str:
    return {0: "Small(1-2)", 1: "Medium(3-5)", 2: "Large(>=6)"}.get(d, "NA")

def pretty_join(tokens):
    """
    更可读的 prefix 展示：
    - 英文 token 之间加空格
    - 中文保持连写
    - 标点紧贴前面
    """
    out = []
    prev_is_en = False
    for tok in tokens:
        is_en = tok.isascii() and any(c.isalpha() for c in tok)
        if not out:
            out.append(tok)
        else:
            if is_punct(tok):
                out.append(tok)  # 标点贴前
            elif is_en:
                out.append((" " if True else "") + tok)
            else:
                # 中文/其它：如果前一个是英文词，前面加空格分隔
                if prev_is_en:
                    out.append(" " + tok)
                else:
                    out.append(tok)
        prev_is_en = is_en
    return "".join(out)

@torch.no_grad()
def predict_on_prefix(model, tokenizer, prefix_tokens):
    enc = tokenizer(
        prefix_tokens,
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt",
    )

    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    # label_pos: prefix 最后一个 word 的首 subword
    word_ids = enc.word_ids(batch_index=0)
    last_word = None
    label_pos = 0
    for j, w in enumerate(word_ids):
        if w is None:
            continue
        if w != last_word:
            last_word = w
            label_pos = j

    logits_sw, logits_dur = model(input_ids, attention_mask)
    ls = logits_sw[0, label_pos, :]
    ld = logits_dur[0, label_pos, :]

    p_switch = torch.softmax(ls, dim=-1)[1].item()
    dur_pred = int(torch.argmax(ld).item())
    return p_switch, dur_pred

def run_demo(text: str, max_steps: int = 200):
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(MODEL_NAME)
    model = XLMRMultiTask(model_name=MODEL_NAME).to(DEVICE)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    model.eval()

    tokens = simple_tokenize(text)
    steps = min(len(tokens), max_steps)

    print("DEVICE:", DEVICE)
    print("CKPT:", CKPT_PATH)
    print("THR_ON / THR_OFF:", SWITCH_THR_ON, "/", SWITCH_THR_OFF)
    print("COOLDOWN_STEPS:", COOLDOWN_STEPS)
    print("Total tokens:", len(tokens))
    print("=" * 100)

    prefix = []
    armed = True
    cooldown = 0

    for i in range(steps):
        tok = tokens[i]
        prefix.append(tok)

        p_sw, d_pred = predict_on_prefix(model, tokenizer, prefix)

        # 冷却期：不触发
        if cooldown > 0:
            cooldown -= 1
            print(f"[t={i:03d}] tok={tok!r:>12}  P(switch)= {p_sw:.3f}")
            continue

        # 过滤：纯标点不触发（避免你看到的逗号/顿号连触发）
        if is_punct(tok):
            print(f"[t={i:03d}] tok={tok!r:>12}  P(switch)= {p_sw:.3f}  (punct)")
            continue

        # hysteresis
        if armed and p_sw >= SWITCH_THR_ON:
            print(f"[t={i:03d}] tok={tok!r:>12}  P(switch)= {p_sw:.3f}  >>> SWITCH  DUR={dur_label_to_str(d_pred)}")
            armed = False
            cooldown = COOLDOWN_STEPS
        else:
            print(f"[t={i:03d}] tok={tok!r:>12}  P(switch)= {p_sw:.3f}")
            if (not armed) and p_sw <= SWITCH_THR_OFF:
                armed = True

    print("=" * 100)
    print("Prefix preview:")
    print(pretty_join(prefix))

if __name__ == "__main__":
    demo_text = (
        "昨天我去參觀了中關村論壇的年會，看到好多機器人在現場幫忙泡茶、帶路，"
        "感覺真的像走進了未來的世界。It was so cool to watch a robot write calligraphy "
        "and even make coffee for the guests, I couldn't believe how advanced everything was!"
    )
    run_demo(demo_text, max_steps=200)
