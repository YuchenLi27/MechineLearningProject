# src/eval_streaming.py
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm
from collections import Counter
import random

from src.model_mt import XLMRMultiTask
from src.dataset_cs import SwitchLinguaCSDataset
from transformers import XLMRobertaTokenizerFast

# 评估配置
MAX_SAMPLES = 5000          # 先 5000，后面可改 20000
POINTS_PER_SAMPLE = 5       # 每条样本随机评估几个位置（自然分布更稳定）
MAX_LEN = 192
SEED = 123

def collate(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
    y_sw = torch.tensor([b["y_sw"] for b in batch], dtype=torch.long)
    y_dur = torch.tensor([b["y_dur"] for b in batch], dtype=torch.long)
    label_pos = torch.tensor([b["label_pos"] for b in batch], dtype=torch.long)
    return dict(input_ids=input_ids, attention_mask=attention_mask, y_sw=y_sw, y_dur=y_dur, label_pos=label_pos)

class EvalStreamingDataset(torch.utils.data.Dataset):
    """
    评估专用：
    - 不做任何 oversampling
    - 每条 base 样本展开成多个 (sample, t) 点
    - t 从“所有可能位置”均匀抽（只要有 t+1 就行）
    - 如果 switch_labels[t] 是 -100（比如涉及 other），我们把它当 0（可选策略）
      目的：不要把大多数位置过滤掉，否则 True switch rate 会被压得极低
    """
    def __init__(self, split="train", max_samples=5000, points_per_sample=5, max_len=192, seed=123):
        self.base = SwitchLinguaCSDataset(split=split, max_samples=max_samples, seed=seed)
        self.tok = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")
        self.max_len = max_len
        self.rng = random.Random(seed)
        self.points = []

        # 预生成 (i, t) 对
        for i in range(len(self.base)):
            ex = self.base[i]
            tokens = ex["tokens"]
            sw = ex["switch_labels"]
            # 所有可能位置：t+1 存在即可
            max_k = min(len(tokens) - 1, len(sw))
            if max_k <= 0:
                continue
            for _ in range(points_per_sample):
                t = self.rng.randrange(0, max_k)
                self.points.append((i, t))

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        i, t = self.points[idx]
        ex = self.base[i]
        tokens = ex["tokens"]
        sw = ex["switch_labels"]
        dur = ex["dur_labels"]

        prefix_tokens = tokens[: t + 1]

        enc = self.tok(
            prefix_tokens,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

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

        raw_sw = int(sw[t]) if t < len(sw) else -100
        y_sw = 0 if raw_sw == -100 else raw_sw

        raw_dur = int(dur[t]) if t < len(dur) else -100
        y_dur = raw_dur  # 仍然只在真实 switch 点有值，否则 -100

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "y_sw": y_sw,
            "y_dur": y_dur,
            "label_pos": label_pos,
        }

@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    ds = EvalStreamingDataset(
        split="train",
        max_samples=MAX_SAMPLES,
        points_per_sample=POINTS_PER_SAMPLE,
        max_len=MAX_LEN,
        seed=SEED,
    )
    dl = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate)

    model = XLMRMultiTask().to(device)
    model.load_state_dict(torch.load("runs/xlmr_streaming.pt", map_location=device))
    model.eval()

    y_true_sw = []
    prob_sw1 = []

    y_true_dur = []
    y_pred_dur = []

    for batch in tqdm(dl):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        y_sw = batch["y_sw"].to(device)
        y_dur = batch["y_dur"].to(device)
        pos = batch["label_pos"].to(device)

        logits_sw, logits_dur = model(input_ids, attention_mask)

        bs = input_ids.size(0)
        idx = torch.arange(bs, device=device)

        ls = logits_sw[idx, pos, :]   # [B,2]
        ld = logits_dur[idx, pos, :]  # [B,3]

        p1 = torch.softmax(ls, dim=-1)[:, 1]
        prob_sw1.extend(p1.detach().cpu().tolist())
        y_true_sw.extend(y_sw.detach().cpu().tolist())

        pred_dur = ld.argmax(dim=-1)
        mask = (y_dur != -100)
        if mask.any():
            y_true_dur.extend(y_dur[mask].detach().cpu().tolist())
            y_pred_dur.extend(pred_dur[mask].detach().cpu().tolist())

    # 计数（关键定位）
    cnt = Counter(y_true_sw)
    true_rate = cnt.get(1, 0) / max(1, len(y_true_sw))

    # 阈值 sweep
    best_f1, best_thr, best_pred_rate = 0.0, 0.5, 0.0
    for thr in [i / 40 for i in range(1, 40)]:
        pred = [1 if p > thr else 0 for p in prob_sw1]
        f1 = f1_score(y_true_sw, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
            best_pred_rate = sum(pred) / len(pred)

    dur_acc = sum(int(a == b) for a, b in zip(y_true_dur, y_pred_dur)) / max(1, len(y_true_dur))

    print("=== Streaming eval (expanded points, natural t sampling) ===")
    print("Total eval points:", len(y_true_sw))
    print("Switch label counts:", dict(cnt))
    print("True switch rate:", round(true_rate, 4))
    print("Best switch F1:", round(best_f1, 4), "at thr=", round(best_thr, 3))
    print("Pred switch rate @best thr:", round(best_pred_rate, 4))
    print("Duration accuracy (argmax):", round(dur_acc, 4))
    print("Duration eval positions:", len(y_true_dur))

if __name__ == "__main__":
    main()
