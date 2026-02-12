import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.model_mt import XLMRMultiTask
from src.dataset_streaming import StreamingPointDataset

def collate(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
    y_sw = torch.tensor([b["y_sw"] for b in batch], dtype=torch.long)
    y_dur = torch.tensor([b["y_dur"] for b in batch], dtype=torch.long)
    label_pos = torch.tensor([b["label_pos"] for b in batch], dtype=torch.long)
    return dict(input_ids=input_ids, attention_mask=attention_mask, y_sw=y_sw, y_dur=y_dur, label_pos=label_pos)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    ds = StreamingPointDataset(split="train", max_samples=5000, max_len=192, seed=0)
    dl = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate)

    model = XLMRMultiTask().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)

    w_sw = torch.tensor([1.0, 4.0], device=device)
    ce_sw = nn.CrossEntropyLoss(weight=w_sw)
    ce_dur = nn.CrossEntropyLoss(ignore_index=-100)

    model.train()
    for step, batch in enumerate(tqdm(dl)):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        y_sw = batch["y_sw"].to(device)
        y_dur = batch["y_dur"].to(device)
        pos = batch["label_pos"].to(device)

        logits_sw, logits_dur = model(input_ids, attention_mask)

        # 只取 label_pos 位置的 logits
        bs = input_ids.size(0)
        idx = torch.arange(bs, device=device)
        ls = logits_sw[idx, pos, :]   # [B,2]
        ld = logits_dur[idx, pos, :]  # [B,3]

        loss_sw = ce_sw(ls, y_sw)

        # duration loss：只在 y_dur != -100 的样本上算，否则设为 0
        mask = (y_dur != -100)
        if mask.any():
            loss_dur = ce_dur(ld[mask], y_dur[mask])
        else:
            loss_dur = torch.zeros((), device=device)

        loss = loss_sw + loss_dur

        # 额外防护：如果出现非有限值，直接跳过该 step（一般不会再发生）
        if not torch.isfinite(loss):
            print("Non-finite loss encountered. Skipping step.")
            opt.zero_grad(set_to_none=True)
            continue

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 50 == 0:
            print(f"step={step} loss={loss.item():.4f} sw={loss_sw.item():.4f} dur={loss_dur.item():.4f}")

        if step >= 800:
            break

    torch.save(model.state_dict(), "runs/xlmr_streaming.pt")
    print("saved runs/xlmr_streaming.pt")

if __name__ == "__main__":
    main()
