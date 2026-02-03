import os, csv, json, argparse, random
from typing import List, Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score, f1_score

try:
    from thop import profile as _thop_profile
    _HAS_THOP = True
except Exception:
    _HAS_THOP = False

import clip

CLASSES: List[str] = [
    "holothurian", "echinus", "scallop", "starfish", "fish",
    "corals", "diver", "cuttlefish", "turtle", "jellyfish"
]
C = len(CLASSES)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray):
    aps = []
    for j in range(y_true.shape[1]):
        if y_true[:, j].sum() > 0:
            aps.append(average_precision_score(y_true[:, j], y_score[:, j]))
    mAP = float(np.mean(aps)) if len(aps) else 0.0
    f1_micro = f1_score(y_true.ravel(), y_pred.ravel(), average='micro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return {'mAP': mAP, 'F1_micro': f1_micro, 'F1_macro': f1_macro}


def _safe_div(num: np.ndarray, den: np.ndarray):
    out = np.zeros_like(num, dtype=np.float64)
    mask = den > 0
    out[mask] = (num[mask] / den[mask]).astype(np.float64)
    return out


def compute_cnki_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    TP = (y_true * y_pred).sum(axis=0).astype(np.float64)
    PP = y_pred.sum(axis=0).astype(np.float64)
    GP = y_true.sum(axis=0).astype(np.float64)

    OP = 0.0 if PP.sum() == 0 else float(TP.sum() / PP.sum())
    OR = 0.0 if GP.sum() == 0 else float(TP.sum() / GP.sum())
    OF1 = 0.0 if (OP + OR) == 0 else float(2 * OP * OR / (OP + OR))

    prec_i = _safe_div(TP, PP)
    rec_i = _safe_div(TP, GP)
    CP = float(prec_i.mean())
    CR = float(rec_i.mean())
    CF1 = 0.0 if (CP + CR) == 0 else float(2 * CP * CR / (CP + CR))

    return {"OP": OP, "OR": OR, "OF1": OF1, "CP": CP, "CR": CR, "CF1": CF1}


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def _num_params(module: nn.Module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def _fmt_m(n): return f"{n / 1e6:.2f}M"
def _fmt_g(n): return f"{n / 1e9:.2f}G"


def report_params_flops(model, adapter=None, device='cuda', img_size=224):
    total_params, trainable_params = _num_params(model)
    inc_total, inc_trainable = 0, 0
    if adapter is not None:
        t, tr = _num_params(adapter)
        inc_total += t
        inc_trainable += tr

    macs = None
    if _HAS_THOP:
        model.visual.eval()
        dummy = torch.randn(1, 3, img_size, img_size, device=device).type(model.dtype)
        try:
            macs, _ = _thop_profile(model.visual, inputs=(dummy,), verbose=False)
        except Exception:
            macs = None

    msg = "[Resource] " \
          f"Params_total={_fmt_m(total_params)}"
    if inc_total > 0:
        msg += f" (+adapter={_fmt_m(inc_total)})"
    msg += f" | Trainable={_fmt_m(trainable_params + inc_trainable)}"
    if macs is not None:
        msg += f" | Visual_MACs={_fmt_g(macs)} | approx_FLOPs={_fmt_g(2 * macs)}/img@{img_size}"
    else:
        msg += " | Visual_MACs=NA (install thop for FLOPs)"
    print(msg)


def read_items(csv_path: str, img_root: str) -> Tuple[List[Tuple[str, np.ndarray]], np.ndarray]:
    items, Ys = [], []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        expect = ["name"] + CLASSES
        if header[:1 + len(CLASSES)] != expect:
            raise ValueError(f"CSV列应为: {expect}\n但实际为: {header[:1 + len(CLASSES)]}")
        for row in reader:
            name = row[0].strip()
            y = np.array([int(v) for v in row[1:1 + len(CLASSES)]], dtype=np.float32)
            path = os.path.join(img_root, name)
            items.append((path, y))
            Ys.append(y)
    Y = np.stack(Ys, axis=0) if Ys else np.zeros((0, C), dtype=np.float32)
    print(f"[CSV] {csv_path}: {len(items)} samples, classes={C}")
    return items, Y


def read_test_dataset(csv_path: str, img_root: str, preprocess):
    if not csv_path:
        return None
    return MultiLabelFixedCSV(csv_path, img_root, preprocess)


class MultiLabelFixedCSV(Dataset):
    def __init__(self, csv_path: str, img_root: str, preprocess):
        super().__init__()
        self.items = []
        self.preprocess = preprocess
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            expect = ["name"] + CLASSES
            if header[:1 + len(CLASSES)] != expect:
                raise ValueError(f"CSV列应为: {expect}\n但实际为: {header[:1 + len(CLASSES)]}")
            for row in reader:
                name = row[0].strip()
                y = np.array([int(v) for v in row[1:1 + len(CLASSES)]], dtype=np.float32)
                path = os.path.join(img_root, name)
                self.items.append((path, y))
        print(f"[Data] {csv_path}: {len(self.items)} samples, classes={C}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p, y = self.items[idx]
        img = Image.open(p).convert('RGB')
        return preprocess_global(img), torch.from_numpy(y)


class MultiLabelFromList(Dataset):
    def __init__(self, items: List[Tuple[str, np.ndarray]], preprocess):
        self.items = items
        self.preprocess = preprocess

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        img = Image.open(p).convert('RGB')
        return self.preprocess(img), torch.from_numpy(y)


class VisualAdapter(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, alpha: float = 0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_dim, bias=False),
        )
        self.alpha = float(alpha)

    def forward(self, img_feat: torch.Tensor, out_dtype: torch.dtype):
        x32 = img_feat.to(torch.float32)
        a32 = self.mlp(x32)
        out32 = self.alpha * a32 + (1.0 - self.alpha) * x32
        return out32.to(out_dtype)


def make_loader_from_items(items: List[Tuple[str, np.ndarray]], preprocess, bs, workers, shuffle):
    ds = MultiLabelFromList(items, preprocess)
    bs = min(bs, len(ds)) if len(ds) > 0 else bs
    return DataLoader(ds, batch_size=max(1, bs), shuffle=shuffle, num_workers=workers, pin_memory=True)


@torch.no_grad()
def infer_logits_clipadapter(model, text_feats, adapter: VisualAdapter, loader, device):
    model.eval()
    adapter.eval()
    all_logits, all_targets = [], []
    for imgs, ys in tqdm(loader, desc="Infer(Adapter)", leave=False):
        imgs = imgs.to(device)
        ys = ys.numpy()
        img_feats = model.encode_image(imgs)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        img_adapt = adapter(img_feats, out_dtype=model.dtype)
        logits = (img_adapt @ text_feats.t()) * model.logit_scale.exp()
        all_logits.append(logits.detach().cpu().numpy())
        all_targets.append(ys)
    return np.concatenate(all_logits, 0), np.concatenate(all_targets, 0)


def tune_thresholds(val_logits: np.ndarray, val_targets: np.ndarray) -> np.ndarray:
    ths = np.zeros(C, dtype=np.float32)
    probs = sigmoid_np(val_logits)
    for j in range(C):
        y = val_targets[:, j]
        if y.sum() == 0:
            ths[j] = 0.5
            continue
        best_f1, best_t = 0.0, 0.5
        for t in np.linspace(0.05, 0.95, 19):
            f1 = f1_score(y, (probs[:, j] > t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        ths[j] = best_t
    return ths


def train_epoch_clipadapter(model, adapter, text_feats, loader, device, optimizer):
    model.eval()
    adapter.train()
    criterion = nn.BCEWithLogitsLoss()
    total = 0.0

    for imgs, ys in tqdm(loader, desc="Train(Adapter)", leave=False):
        imgs = imgs.to(device)
        ys = ys.to(device)
        with torch.no_grad():
            img_feats = model.encode_image(imgs)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

        optimizer.zero_grad(set_to_none=True)
        img_adapt = adapter(img_feats, out_dtype=model.dtype)
        logits = (img_adapt @ text_feats.t()) * model.logit_scale.exp()
        loss = criterion(logits, ys)
        loss.backward()
        optimizer.step()
        total += loss.item() * imgs.size(0)

    return total / len(loader.dataset)


def single_split_train_validate(model, preprocess, args,
                                train_items, val_items,
                                text_feats):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_train = len(train_items)
    if n_train < 50:
        hidden_dim = min(32, args.adapter_hidden)
    elif n_train < 200:
        hidden_dim = min(64, args.adapter_hidden)
    else:
        hidden_dim = args.adapter_hidden
    print(f"[Adapter] n_train={n_train}, use hidden_dim={hidden_dim} (orig={args.adapter_hidden})")

    patience = args.patience
    if n_train < 50:
        patience = max(patience, 10)
    elif n_train < 200:
        patience = max(patience, 8)
    print(f"[Train] Use early-stopping patience={patience}")

    with torch.no_grad():
        dummy_img = (np.random.rand(224, 224, 3) * 255).astype('uint8')
        dummy = preprocess(Image.fromarray(dummy_img)).unsqueeze(0).to(device)
        img_feat_dim = model.encode_image(dummy).shape[-1]

    adapter = VisualAdapter(img_feat_dim, hidden_dim=hidden_dim,
                            alpha=args.adapter_alpha).to(device)

    params = [p for p in adapter.parameters() if p.requires_grad]
    optim = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)

    train_loader = make_loader_from_items(train_items, preprocess, args.bs, args.workers, shuffle=True)
    val_loader = make_loader_from_items(val_items, preprocess, args.bs, args.workers, shuffle=False)

    best_map, best_epoch, stale = -1.0, 0, 0
    best_ths = np.full(C, 0.5, np.float32)
    ckpt_path = os.path.join(args.ckpt_dir, f"clipadapter_best.pt")

    for ep in range(1, args.epochs + 1):
        train_loss = train_epoch_clipadapter(
            model, adapter, text_feats, train_loader, device, optim
        )

        with torch.no_grad():
            val_logits, val_targets = infer_logits_clipadapter(model, text_feats, adapter, val_loader, device)
            ths = tune_thresholds(val_logits, val_targets) if args.tune_thresh else np.full(C, 0.5, np.float32)
            probs = sigmoid_np(val_logits)
            pred = (probs > ths[None, :]).astype(int)
            m = compute_metrics(val_targets, probs, pred)
            cur_map = m['mAP']

        if ep % max(1, args.epochs // 10) == 0 or ep == 1:
            print(f"[Single] Epoch {ep:03d}/{args.epochs}  loss={train_loss:.4f}  "
                  f"Val mAP={cur_map:.4f} F1micro={m['F1_micro']:.4f} F1macro={m['F1_macro']:.4f}")

        if cur_map > best_map + 1e-6:
            best_map, best_epoch, best_ths, stale = cur_map, ep, ths, 0
            ensure_dir(os.path.dirname(ckpt_path))
            payload = {
                "adapter_state": adapter.state_dict(),
                "extra": {"epoch": ep, "mAP": best_map, "ths": best_ths}
            }
            torch.save(payload, ckpt_path)
        else:
            stale += 1
            if stale >= patience:
                print(f"[Single] Early stop at epoch {ep} (best epoch {best_epoch}, best mAP {best_map:.4f})")
                break

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    adapter.load_state_dict(payload["adapter_state"])
    best_ths = payload.get("extra", {}).get("ths", best_ths)

    return adapter, best_ths, best_epoch, best_map


def build_multilabel_fewshot_indices(Y: np.ndarray, shots: int, seed: int = 42):
    rng = np.random.RandomState(seed)
    N, C_ = Y.shape
    pos_pool = [set(np.where(Y[:, c] > 0.5)[0].tolist()) for c in range(C_)]
    need = [shots] * C_
    chosen = set()

    for c in range(C_):
        if len(pos_pool[c]) < shots:
            print(f"[FewShot] WARN: class {c} only has {len(pos_pool[c])} positives < {shots}")

    remain = set(range(C_))
    while len(remain) > 0:
        cover_count = np.zeros(N, dtype=np.int32)
        for c in list(remain):
            for i in pos_pool[c]:
                cover_count[i] += 1
        if cover_count.max() == 0:
            break
        best = np.where(cover_count == cover_count.max())[0]
        i = int(rng.choice(best))
        chosen.add(i)
        hit = [c for c in list(remain) if i in pos_pool[c]]
        for c in hit:
            need[c] -= 1
            if need[c] <= 0:
                remain.discard(c)
        for c in range(C_):
            if i in pos_pool[c]:
                pos_pool[c].discard(i)

    for c in range(C_):
        while need[c] > 0 and len(pos_pool[c]) > 0:
            i = int(pos_pool[c].pop())
            chosen.add(i)
            need[c] -= 1

    chosen = sorted(list(chosen))
    print(f"[FewShot] Selected {len(chosen)} images to satisfy >={shots} positives per class (best-effort).")
    return np.array(chosen, dtype=int)


@torch.no_grad()
def build_text_features(model, device):
    texts = [f"a photo of a {c}." for c in CLASSES]
    tokenized = clip.tokenize(texts, truncate=True).to(device)
    text_feats = model.encode_text(tokenized)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    return text_feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img_root', type=str, default='/path/to/images')
    ap.add_argument('--csv', type=str, default='multilabel_train.csv')
    ap.add_argument('--val_csv', type=str, default='multilabel_val.csv')
    ap.add_argument('--test_csv', type=str, default='multilabel_test.csv')
    ap.add_argument('--backbone', type=str, default='RN50')
    ap.add_argument('--mode', type=str, default='clipadapter',
                    help="为兼容旧命令行，仅接受 'clipadapter'")
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--tune_thresh', action='store_true')
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--ckpt_dir', type=str, default='checkpoints')
    ap.add_argument('--shots', type=int, default=16,
                    help="few-shot per-class positives on TRAIN only (0=full train)")
    ap.add_argument('--adapter_hidden', type=int, default=256, help="视觉 Adapter 瓶颈维度 r")
    ap.add_argument('--adapter_alpha', type=float, default=0.2, help="残差系数 α，对应 f* = α Av(f) + (1-α) f")
    ap.add_argument('--min_train', type=int, default=0,
                    help="few-shot 模式下，至少使用多少张训练图像（通过随机补样扩展）")
    args = ap.parse_args()

    if args.mode != 'clipadapter':
        raise ValueError("当前脚本仅支持 --mode clipadapter，请不要传其他模式。")

    args.img_root = 'images'

    if not args.val_csv:
        raise ValueError("必须提供 --val_csv 作为验证集。")

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device} | Classes={CLASSES} | Patience={args.patience} | Ckpt={args.ckpt_dir}")

    model, preprocess = clip.load(args.backbone, device=device, jit=False)
    for p in model.parameters():
        p.requires_grad = False

    global preprocess_global
    preprocess_global = preprocess

    text_feats = build_text_features(model, device)

    train_items, Y = read_items(args.csv, args.img_root)

    if args.shots and args.shots > 0:
        few_idx = build_multilabel_fewshot_indices(Y, shots=args.shots, seed=args.seed)

        if len(few_idx) < args.min_train:
            all_idx = np.arange(len(Y))
            remain_idx = np.setdiff1d(all_idx, few_idx)
            extra_needed = min(args.min_train - len(few_idx), len(remain_idx))
            if extra_needed > 0:
                extra = np.random.choice(remain_idx, size=extra_needed, replace=False)
                few_idx = np.concatenate([few_idx, extra])
                few_idx = np.unique(few_idx)
            print(f"[FewShot] Expand to {len(few_idx)} samples (min_train={args.min_train}).")

        train_items = [train_items[i] for i in few_idx]
        Y = Y[few_idx]
        print(f"[FewShot] Using subset: {len(train_items)} samples for TRAIN (val/test 不动).")

    val_items, _ = read_items(args.val_csv, args.img_root)
    adapter, best_ths, best_epoch, best_map = single_split_train_validate(
        model, preprocess, args, train_items, val_items, text_feats
    )

    test_ds = read_test_dataset(args.test_csv, args.img_root, preprocess)
    if test_ds is not None:
        test_loader = DataLoader(test_ds, batch_size=args.bs, shuffle=False,
                                 num_workers=args.workers, pin_memory=True)
        with torch.no_grad():
            test_logits, test_targets = infer_logits_clipadapter(model, text_feats, adapter, test_loader, device)

            probs = sigmoid_np(test_logits)
            ths = best_ths if args.tune_thresh else np.full(C, 0.5, np.float32)
            pred = (probs > ths[None, :]).astype(int)

            m = compute_metrics(test_targets, probs, pred)
            print(f"[Test] " + json.dumps({"mAP": m["mAP"]}, ensure_ascii=False))
            cnki = compute_cnki_metrics(test_targets, pred)
            print(f"[Test-CNKI] {json.dumps(cnki, ensure_ascii=False)}")

    report_params_flops(model, adapter=adapter, device=device, img_size=224)


if __name__ == "__main__":
    main()