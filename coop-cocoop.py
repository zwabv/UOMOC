import os, csv, json, argparse, random
from typing import List, Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score, f1_score
import clip

# =============== 固定类别（与CSV列顺序一致） ===============
CLASSES: List[str] = [
    "holothurian", "echinus", "scallop", "starfish", "fish",
    "corals", "diver", "cuttlefish", "turtle", "jellyfish"
]
C = len(CLASSES)


# ---------------- Utils ----------------
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

def report_params_flops(model, pl=None, cocoop_wrap=None, device='cuda', img_size=224):
    total_params, trainable_params = _num_params(model)
    inc_total, inc_trainable = 0, 0
    if pl is not None:
        t, tr = _num_params(pl)
        inc_total += t
        inc_trainable += tr
    if cocoop_wrap is not None:
        t, tr = _num_params(cocoop_wrap.metanet)
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
        msg += f" (+prompt/meta={_fmt_m(inc_total)})"
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
    if not csv_path: return None
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



class PromptLearner(nn.Module):
    def __init__(self, classnames: List[str], clip_model, ctx_len=4, csc=False, place='mid', init='phrase'):
        super().__init__()
        assert place in ['end', 'mid']
        assert init in ['phrase', 'randn']
        self.classnames = classnames
        self.ctx_len = ctx_len
        self.csc = bool(csc)
        self.place = place
        self.init = init

        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

        texts = [f"a photo of a {c}." for c in classnames]
        tokenized = clip.tokenize(texts, truncate=True)
        self.register_buffer("tokenized", tokenized)

        with torch.no_grad():
            class_embeds = self.token_embedding(self.tokenized.to(next(clip_model.parameters()).device)).type(dtype)
        self.register_buffer("class_embeds", class_embeds)

        if self.csc:
            self.ctx = nn.Parameter(torch.empty(len(classnames), ctx_len, ctx_dim, dtype=dtype))
        else:
            self.ctx = nn.Parameter(torch.empty(ctx_len, ctx_dim, dtype=dtype))

        if self.init == 'phrase':
            ctx_init = self._init_ctx_with_phrase(clip_model, ctx_len, ctx_dim).type(dtype)
            if self.csc:
                self.ctx.data.copy_(ctx_init.unsqueeze(0).repeat(len(classnames), 1, 1))
            else:
                self.ctx.data.copy_(ctx_init)
        else:
            nn.init.normal_(self.ctx, std=0.02)

    @torch.no_grad()
    def _init_ctx_with_phrase(self, clip_model, ctx_len, ctx_dim):
        phrase = "a photo of a"
        tok = clip.tokenize([phrase], truncate=True).to(next(clip_model.parameters()).device)
        emb = clip_model.token_embedding(tok)[0]
        eos = (tok[0] == 49407).nonzero(as_tuple=False)[0].item()
        core = emb[1:1 + 4, :]
        if core.shape[0] < 4:
            extra = emb[1 + core.shape[0]:1 + 4, :]
            core = torch.cat([core, extra], dim=0)
        if ctx_len == 4:
            out = core
        elif ctx_len < 4:
            out = core[:ctx_len, :]
        else:
            reps = (ctx_len + 3) // 4
            out = core.repeat((reps, 1))[:ctx_len, :]
        return out.clone()

    def _compose_prompt(self, i: int):
        emb = self.class_embeds[i].clone()
        ctx = self.ctx[i] if self.csc else self.ctx
        tok = self.tokenized[i]
        eos = (tok == 49407).nonzero(as_tuple=False)[0].item()
        if self.place == 'end':
            body_len = eos - 1
            new = torch.zeros_like(emb)
            new[:body_len, :] = emb[:body_len, :]
            insert_end = body_len + min(self.ctx_len, 77 - body_len - 1)
            new[body_len:insert_end, :] = ctx[:insert_end - body_len, :]
            new[insert_end, :] = emb[eos, :]
            return new
        else:
            prefix_len = min(1 + 5, eos)
            new = torch.zeros_like(emb)
            new[:prefix_len, :] = emb[:prefix_len, :]
            insert_end = prefix_len + min(self.ctx_len, 77 - prefix_len - 1)
            new[prefix_len:insert_end, :] = ctx[:insert_end - prefix_len, :]
            tail_room = 77 - insert_end - 1
            tail = emb[prefix_len:prefix_len + tail_room, :]
            new[insert_end:insert_end + tail.shape[0], :] = tail
            new[insert_end + tail.shape[0], :] = emb[eos, :]
            return new

    def forward_text_features(self, clip_model, device):
        x_list = []
        for i in range(len(self.classnames)):
            emb = self._compose_prompt(i).to(device)
            x = emb.unsqueeze(0) + self.positional_embedding.to(device).unsqueeze(0)
            x = x.type(clip_model.dtype)
            x = x.permute(1, 0, 2)
            x = clip_model.transformer(x)
            x = x.permute(1, 0, 2)
            x = clip_model.ln_final(x).type(clip_model.dtype)
            eos = (self.tokenized[i].to(device) == 49407).nonzero(as_tuple=False)[0].item()
            x = x[0, eos, :] @ clip_model.text_projection
            x_list.append(x)
        x = torch.stack(x_list, dim=0)
        return x / x.norm(dim=-1, keepdim=True)


    def forward_text_features_with_pi(self, clip_model, device, pi: torch.Tensor):
        x_list = []
        for i in range(len(self.classnames)):
            base = self._compose_prompt(i).to(device)
            tok = self.tokenized[i].to(device)
            eos = (tok == 49407).nonzero(as_tuple=False)[0].item()
            if self.place == 'end':
                body_len = eos - 1
                start = body_len
                end = min(body_len + self.ctx_len, 77 - 1)
            else:
                prefix_len = min(1 + 5, eos)
                start = prefix_len
                end = min(prefix_len + self.ctx_len, 77 - 1)
            # pi 已在外部转为 clip dtype
            base[start:end, :] = base[start:end, :] + pi.unsqueeze(0).expand(end - start, -1)

            x = base.unsqueeze(0) + self.positional_embedding.to(device).unsqueeze(0)
            x = x.type(clip_model.dtype)
            x = x.permute(1, 0, 2)
            x = clip_model.transformer(x)
            x = x.permute(1, 0, 2)
            x = clip_model.ln_final(x).type(clip_model.dtype)
            x = x[0, eos, :] @ clip_model.text_projection
            x_list.append(x)
        x = torch.stack(x_list, dim=0)
        return x / x.norm(dim=-1, keepdim=True)


class MetaNet(nn.Module):
    def __init__(self, in_dim: int, ctx_dim: int):
        super().__init__()
        bottleneck = max(1, in_dim // 16)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, bottleneck),
            nn.ReLU(True),
            nn.Linear(bottleneck, ctx_dim)
        )

    def forward(self, img_feat: torch.Tensor):
        return self.mlp(img_feat)


class CoCoOpWrapper(nn.Module):
    def __init__(self, prompt_learner: PromptLearner, clip_model, img_feat_dim: int):
        super().__init__()
        self.pl = prompt_learner
        self.clip_model = clip_model
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.metanet = MetaNet(img_feat_dim, ctx_dim)


    def conditional_text_features(self, img_feat: torch.Tensor, device):
        meta_dtype = self.metanet.mlp[0].weight.dtype
        img_feat_meta = img_feat.to(meta_dtype)
        pi = self.metanet(img_feat_meta)
        pi = pi.to(self.clip_model.dtype)
        return self.pl.forward_text_features_with_pi(self.clip_model, device, pi)


# ---------------- Train / Eval ----------------
def make_loader_from_items(items: List[Tuple[str, np.ndarray]], preprocess, bs, workers, shuffle):
    ds = MultiLabelFromList(items, preprocess)
    bs = min(bs, len(ds)) if len(ds) > 0 else bs
    return DataLoader(ds, batch_size=max(1, bs), shuffle=shuffle, num_workers=workers, pin_memory=True)


@torch.no_grad()
def infer_logits(model, text_feats, loader, device):
    model.eval()
    all_logits, all_targets = [], []
    for imgs, ys in tqdm(loader, desc="Infer", leave=False):
        imgs = imgs.to(device)
        ys = ys.numpy()
        img_feats = model.encode_image(imgs)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        logits = (img_feats @ text_feats.t()) * model.logit_scale.exp()
        all_logits.append(logits.cpu().numpy())
        all_targets.append(ys)
    return np.concatenate(all_logits, 0), np.concatenate(all_targets, 0)


@torch.no_grad()
def infer_logits_cocoop(model, pl: PromptLearner, cocoop_wrap: CoCoOpWrapper, loader, device):
    model.eval()
    all_logits, all_targets = [], []
    for imgs, ys in tqdm(loader, desc="Infer(CoCoOp)", leave=False):
        imgs = imgs.to(device)
        ys_np = ys.numpy()
        with torch.no_grad():
            img_feats = model.encode_image(imgs)  # [B, d] (clip dtype)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        batch_logits = []
        for b in range(img_feats.size(0)):
            tf = cocoop_wrap.conditional_text_features(img_feats[b], device)  # [C, d]
            logit_b = (img_feats[b:b + 1] @ tf.t()) * model.logit_scale.exp()  # [1, C]
            batch_logits.append(logit_b)
        batch_logits = torch.cat(batch_logits, dim=0)
        all_logits.append(batch_logits.cpu().numpy())
        all_targets.append(ys_np)
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


def train_epoch(model, prompt_module, loader, device, optimizer, mode='coop', cocoop_wrap: CoCoOpWrapper = None):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total = 0.0
    for imgs, ys in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(device)
        ys = ys.to(device)

        with torch.no_grad():
            img_feats = model.encode_image(imgs)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

        optimizer.zero_grad(set_to_none=True)

        if mode == 'coop':
            text_feats = prompt_module.forward_text_features(model, device)
            logits = (img_feats @ text_feats.t()) * model.logit_scale.exp()
            loss = criterion(logits, ys)
            loss.backward()
            optimizer.step()
            total += loss.item() * imgs.size(0)

        else:
            losses = []
            for b in range(imgs.size(0)):
                tf_b = cocoop_wrap.conditional_text_features(img_feats[b], device)  # [C, d]
                logit_b = (img_feats[b:b + 1] @ tf_b.t()) * model.logit_scale.exp()
                loss_b = criterion(logit_b, ys[b:b + 1])
                losses.append(loss_b)
            if not losses:
                continue
            loss_sum = torch.stack(losses).sum()
            loss_sum.backward()
            optimizer.step()
            total += float(loss_sum.item())

    return total / len(loader.dataset)


def single_split_train_validate(model, preprocess, args,
                                train_items, val_items,
                                mode='coop'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pl = PromptLearner(CLASSES, model, ctx_len=args.ctx_len, csc=bool(args.csc), place=args.place, init='phrase').to(
        device)
    params = [p for p in pl.parameters() if p.requires_grad]
    cocoop_wrap = None
    if mode == 'cocoop':
        with torch.no_grad():
            dummy = preprocess(Image.fromarray((np.random.rand(224, 224, 3) * 255).astype('uint8'))).unsqueeze(0).to(
                device)
            img_feat_dim = model.encode_image(dummy).shape[-1]
        cocoop_wrap = CoCoOpWrapper(pl, model, img_feat_dim).to(device)
        params += [p for p in cocoop_wrap.metanet.parameters() if p.requires_grad]
    optim = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)

    train_loader = make_loader_from_items(train_items, preprocess, args.bs, args.workers, shuffle=True)
    val_loader = make_loader_from_items(val_items, preprocess, args.bs, args.workers, shuffle=False)

    best_map, best_epoch, stale = -1.0, 0, 0
    best_ths = np.full(C, 0.5, np.float32)
    ckpt_path = os.path.join(args.ckpt_dir, f"single_best.pt")

    for ep in range(1, args.epochs + 1):
        train_loss = train_epoch(model, pl, train_loader, device, optim, mode=mode, cocoop_wrap=cocoop_wrap)
        with torch.no_grad():
            if mode == 'cocoop':
                val_logits, val_targets = infer_logits_cocoop(model, pl, cocoop_wrap, val_loader, device)
            else:
                text_feats = pl.forward_text_features(model, device)
                val_logits, val_targets = infer_logits(model, text_feats, val_loader, device)

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
            payload = {"prompt_state": pl.state_dict()}
            if cocoop_wrap is not None:
                payload["metanet_state"] = cocoop_wrap.metanet.state_dict()
            payload["extra"] = {"epoch": ep, "mAP": best_map, "ths": best_ths}
            torch.save(payload, ckpt_path)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"[Single] Early stop at epoch {ep} (best epoch {best_epoch}, best mAP {best_map:.4f})")
                break

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pl.load_state_dict(payload["prompt_state"])
    if ("metanet_state" in payload) and (cocoop_wrap is not None):
        cocoop_wrap.metanet.load_state_dict(payload["metanet_state"])
    best_ths = payload.get("extra", {}).get("ths", best_ths)

    return pl, cocoop_wrap, best_ths, best_epoch, best_map


# ---------------- Few-shot for Multilabel ----------------
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
    return chosen


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img_root', type=str, default='/path/to/images')
    ap.add_argument('--csv', type=str, default='multilabel_train.csv')
    ap.add_argument('--val_csv', type=str, required=True, help="必须提供验证集CSV路径（单次验证）")
    ap.add_argument('--test_csv', type=str, default='multilabel_test.csv')
    ap.add_argument('--backbone', type=str, default='ViT-B/32')
    ap.add_argument('--mode', type=str, default='cocoop', choices=['coop', 'cocoop'])
    ap.add_argument('--ctx_len', type=int, default=4)
    ap.add_argument('--place', type=str, default='mid', choices=['end', 'mid'])
    ap.add_argument('--csc', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--bs', type=int, default=1)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--tune_thresh', action='store_true')
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--ckpt_dir', type=str, default='checkpoints')
    ap.add_argument('--shots', type=int, default=0,
                    help="few-shot per-class positives on TRAIN only (0=full train)")
    args = ap.parse_args()
    args.img_root = 'images'

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device} | Classes={CLASSES} | Patience={args.patience} | Ckpt={args.ckpt_dir}")


    model, preprocess = clip.load(args.backbone, device=device, jit=False)
    for p in model.parameters():
        p.requires_grad = False

    global preprocess_global
    preprocess_global = preprocess


    train_items, Y = read_items(args.csv, args.img_root)
    if args.shots and args.shots > 0:
        few_idx = build_multilabel_fewshot_indices(Y, shots=args.shots, seed=args.seed)
        train_items = [train_items[i] for i in few_idx]
        Y = Y[few_idx]
        print(f"[FewShot] Using subset: {len(train_items)} samples for TRAIN (val/test 不动).")


    val_items, _ = read_items(args.val_csv, args.img_root)
    pl, cocoop_wrap, best_ths, best_epoch, best_map = single_split_train_validate(
        model, preprocess, args, train_items, val_items, mode=args.mode
    )
    test_ds = read_test_dataset(args.test_csv, args.img_root, preprocess)
    if test_ds is not None:
        test_loader = DataLoader(test_ds, batch_size=args.bs, shuffle=False,
                                 num_workers=args.workers, pin_memory=True)
        with torch.no_grad():
            if args.mode == 'cocoop':
                test_logits, test_targets = infer_logits_cocoop(model, pl, cocoop_wrap, test_loader, device)
            else:
                text_feats = pl.forward_text_features(model, device)
                test_logits, test_targets = infer_logits(model, text_feats, test_loader, device)

            probs = sigmoid_np(test_logits)
            ths = best_ths if args.tune_thresh else np.full(C, 0.5, np.float32)
            pred = (probs > ths[None, :]).astype(int)

            m = compute_metrics(test_targets, probs, pred)
            print(f"[Test] " + json.dumps({"mAP": m["mAP"]}, ensure_ascii=False))
            cnki = compute_cnki_metrics(test_targets, pred)
            print(f"[Test-CNKI] {json.dumps(cnki, ensure_ascii=False)}")

        report_params_flops(model, pl, cocoop_wrap, device=device, img_size=224)


if __name__ == "__main__":
    main()