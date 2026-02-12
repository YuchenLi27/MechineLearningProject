# src/dataset_streaming.py
import random
from torch.utils.data import Dataset
from transformers import XLMRobertaTokenizerFast

from src.dataset_cs import SwitchLinguaCSDataset

class StreamingPointDataset(Dataset):
    """
    Pointwise streaming dataset:
      - sample a position t in a sequence
      - input: prefix tokens[0..t]
      - labels: switch_next[t] (0/1), duration3[t] (0/1/2 or -100)

    p_switch:
      - training: set high (e.g., 0.7) to oversample switch points
      - evaluation: set 0.0 to sample uniformly from valid positions (natural)
    """
    def __init__(self, split="train", max_samples=5000, max_len=192, seed=0, p_switch=0.0):
        self.base = SwitchLinguaCSDataset(split=split, max_samples=max_samples, seed=seed)
        self.tok = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")
        self.max_len = max_len
        self.rng = random.Random(seed)
        self.p_switch = float(p_switch)

    def __len__(self):
        return len(self.base)

    def _pick_t(self, tokens, sw):
        # valid positions are those with defined switch label (not -100)
        max_k = min(len(tokens) - 1, len(sw))
        max_k = min(len(tokens) - 1, len(sw))  # positions with t+1
        valid = list(range(max_k))

        if not valid:
            return 0

        pos_switch = [k for k in valid if sw[k] == 1]

        # IMPORTANT FIX:
        # If p_switch == 0.0, do NOT bias to non-switch.
        # Sample uniformly from valid positions to reflect natural distribution.
        if self.p_switch <= 0.0 or not pos_switch:
            return self.rng.choice(valid)

        # Otherwise: oversample switch points with probability p_switch,
        # else sample uniformly from valid (not "only non-switch")
        if self.rng.random() < self.p_switch:
            return self.rng.choice(pos_switch)
        return self.rng.choice(valid)

    def __getitem__(self, i):
        ex = self.base[i]
        tokens = ex["tokens"]
        sw = ex["switch_labels"]
        dur = ex["dur_labels"]

        t = self._pick_t(tokens, sw)
        prefix_tokens = tokens[: t + 1]

        enc = self.tok(
            prefix_tokens,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        # label position: first subword of the last word in the prefix
        word_ids = enc.word_ids(batch_index=0)
        last_word = None
        label_pos = 0
        for j, w in enumerate(word_ids):
            if w is None:
                continue
            if w != last_word:
                last_word = w
                label_pos = j

        raw = int(sw[t])
        y_sw = 0 if raw == -100 else raw
        y_dur = int(dur[t]) if (t < len(dur) and dur[t] != -100) else -100

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "y_sw": y_sw,
            "y_dur": y_dur,
            "label_pos": label_pos,
        }
