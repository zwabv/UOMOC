import os
import json
import random
import torch
import clip
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_curve, auc, precision_score, recall_score, f1_score, accuracy_score
import argparse

try:
    from thop import profile as _thop_profile
    _HAS_THOP = True
except Exception:
    _HAS_THOP = False

# 固定随机种子
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


FEWSHOT = True
SHOTS   = 16
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.float()
os.makedirs("results", exist_ok=True)


class_list = [
    "holothurian", "echinus", "scallop", "starfish", "fish",
    "corals", "diver", "cuttlefish", "turtle", "jellyfish"
]
num_classes = len(class_list)


def _load_json_list(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到数据文件: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

train_json_path = "multilabel_train.json"
val_json_path   = "multilabel_val.json"
test_json_path  = "multilabel_test.json"

train_all = _load_json_list(train_json_path)
val_all   = _load_json_list(val_json_path)
test_all  = _load_json_list(test_json_path)
image_folder = "images"


UNDERWATER_TEMPLATES = [
    "a photo of a {}.",
    "an underwater photo of a {}.",
    "a marine image of a {}.",
]

def build_underwater_text_features(class_list, clip_model, device):
    all_feats = []
    with torch.no_grad():
        for cname in class_list:
            feats = []
            for tpl in UNDERWATER_TEMPLATES:
                prompt = tpl.format(cname)
                tokens = clip.tokenize([prompt]).to(device)
                feat = clip_model.encode_text(tokens)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                feats.append(feat)
            cls_feat = torch.stack(feats, dim=0).mean(dim=0)
            all_feats.append(cls_feat.squeeze(0))
    text_feats = torch.stack(all_feats, dim=0)
    return text_feats.float()

text_features = build_underwater_text_features(class_list, clip_model, device).to(device)

class CoCoOpPromptAdapter(nn.Module):
    def __init__(self, base_text_all, hidden_dim=256, learnable_base=True):
        super().__init__()
        if learnable_base:
            self.base_text_all = nn.Parameter(base_text_all.clone())
        else:
            self.register_buffer("base_text_all", base_text_all)

        C, emb_dim = base_text_all.shape
        self.meta_net = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, C)
        )

    def forward(self, img_feat):
        base = self.base_text_all
        base = base / (base.norm(dim=-1, keepdim=True))

        B, C = img_feat.size(0), base.size(0)
        logits_alpha = self.meta_net(img_feat.float())
        alpha = torch.softmax(logits_alpha, dim=-1)
        text = alpha @ base
        text = text / (text.norm(dim=-1, keepdim=True))
        return text

feat_cache_path = "clip_feats_cache.pt"

def _collect_all_names(*lists):
    names = []
    for L in lists:
        for item in L:
            names.append(item["name"])
    return sorted(list(set(names)))

all_names = _collect_all_names(train_all, val_all, test_all)

if os.path.exists(feat_cache_path):
    print(f"加载特征缓存: {feat_cache_path}")
    all_feats = torch.load(feat_cache_path)
    all_feats = {k: v.float() for k, v in all_feats.items() if k in all_names}
    missing = [n for n in all_names if n not in all_feats]
    if len(missing) > 0:
        print(f"补齐{len(missing)}个缺失特征")
        for img_name in tqdm(missing, desc="补齐特征"):
            img_path = os.path.join(image_folder, img_name)
            try:
                image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
                with torch.no_grad():
                    feat = clip_model.encode_image(image)
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                all_feats[img_name] = feat.cpu()
            except Exception as e:
                print(f"图像{img_name}提取失败: {e}")
        torch.save(all_feats, feat_cache_path)
else:
    print("提取CLIP图像特征")
    all_feats = {}
    for img_name in tqdm(all_names, desc="提取特征"):
        img_path = os.path.join(image_folder, img_name)
        try:
            image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = clip_model.encode_image(image)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            all_feats[img_name] = feat.cpu().float()
        except Exception as e:
            print(f"图像{img_name}提取失败: {e}")
    torch.save(all_feats, feat_cache_path)


class FeatureDataset(Dataset):
    def __init__(self, data, class_list, all_feats):
        self.data = data
        self.class_list = class_list
        self.all_feats = all_feats
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        img_feat = self.all_feats[item["name"]].squeeze(0)
        label = torch.tensor([item[c] for c in self.class_list], dtype=torch.float32)
        return img_feat, label, item["name"]

def _build_loader(data, batch_size, shuffle):
    ds = FeatureDataset(data, class_list, all_feats)
    bs = min(batch_size, len(ds)) if len(ds) > 0 else batch_size
    return DataLoader(ds, batch_size=max(1, bs), shuffle=shuffle, num_workers=4, pin_memory=True)


def calculate_ap(gt, pred):
    precision, recall, _ = precision_recall_curve(gt, pred)
    return auc(recall, precision)

def calculate_mAP(label_data, pred_data):
    ap_list = []
    for cname in class_list:
        gt = [int(sample[cname]) for sample in pred_data]
        pred = [sample[cname] for sample in pred_data]
        ap_list.append(calculate_ap(gt, pred))
    return float(np.mean(ap_list)) if len(ap_list) > 0 else 0.0

def compute_metrics(label_data, pred_data, threshold=0.5):
    y_true, y_pred = [], []
    for i in range(len(label_data)):
        y_true.append([int(label_data[i][c]) for c in class_list])
        y_pred.append([1 if pred_data[i][c] >= threshold else 0 for c in class_list])
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc = accuracy_score(y_true.ravel(), y_pred.ravel())
    prec = precision_score(y_true.ravel(), y_pred.ravel(), zero_division=0)
    rec = recall_score(y_true.ravel(), y_pred.ravel(), zero_division=0)
    f1 = f1_score(y_true.ravel(), y_pred.ravel(), zero_division=0)
    return acc, prec, rec, f1

def tune_thresholds_from_val(val_labels, val_probs, class_list):
    C = len(class_list)
    y_true = np.array([[int(x[c]) for c in class_list] for x in val_labels], dtype=int)
    y_prob = np.array([[x[c] for c in class_list] for x in val_probs], dtype=float)
    ths = np.full(C, 0.5, dtype=float)
    grid = np.linspace(0.05, 0.95, 19)
    for j in range(C):
        best_f1, best_t = 0.0, 0.5
        yj = y_true[:, j]
        pj = y_prob[:, j]
        if yj.sum() == 0:
            continue
        for t in grid:
            f1 = f1_score(yj, (pj >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
            ths[j] = best_t
    return ths

def compute_pos_weight(train_labels, class_list):
    y = np.array([[int(x[c]) for c in class_list] for x in train_labels], dtype=int)
    P = y.sum(axis=0).astype(float)
    N = (y.shape[0] - P).astype(float)
    pw = N / np.clip(P, 1.0, None)
    return torch.tensor(pw, dtype=torch.float32)

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
    rec_i  = _safe_div(TP, GP)
    CP = float(prec_i.mean())
    CR = float(rec_i.mean())
    CF1 = 0.0 if (CP + CR) == 0 else float(2 * CP * CR / (CP + CR))
    return {"OP": OP, "OR": OR, "OF1": OF1, "CP": CP, "CR": CR, "CF1": CF1}


def build_fewshot_subset(train_list, class_list, shots=8, seed=42):
    rng = random.Random(seed)
    pos_pool = {c: [i for i, x in enumerate(train_list) if int(x[c]) == 1] for c in class_list}
    need = {c: shots for c in class_list}
    chosen = set()
    for c in class_list:
        if len(pos_pool[c]) < shots:
            print(f"[FewShot] 类别{c}正样本数{len(pos_pool[c])}<{shots}")
    remain = set(class_list)
    while len(remain) > 0:
        cover_count = np.zeros(len(train_list), dtype=np.int32)
        for c in list(remain):
            for i in pos_pool[c]:
                cover_count[i] += 1
        if cover_count.max() == 0:
            break
        best_idx_candidates = np.where(cover_count == cover_count.max())[0].tolist()
        i = rng.choice(best_idx_candidates)
        chosen.add(i)
        hit_classes = [c for c in list(remain) if i in pos_pool[c]]
        for c in hit_classes:
            need[c] -= 1
            if need[c] <= 0:
                remain.discard(c)
        for c in class_list:
            if i in pos_pool[c]:
                pos_pool[c] = [k for k in pos_pool[c] if k != i]
    for c in class_list:
        while need[c] > 0 and len(pos_pool[c]) > 0:
            i = pos_pool[c].pop()
            chosen.add(i)
            need[c] -= 1
    chosen = sorted(list(chosen))
    fewshot_list = [train_list[i] for i in chosen]
    print(f"[FewShot] 选了{len(fewshot_list)}张图满足每类至少{shots}个正样本")
    return fewshot_list

def _num_params(module: nn.Module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable
def _fmt_m(n): return f"{n/1e6:.2f}M"
def _fmt_g(n): return f"{n/1e9:.2f}G"

def report_params_flops_adapter(clip_model, adapter_m):
    total_params, _ = _num_params(clip_model)
    adp_total, adp_trainable = _num_params(adapter_m)
    macs = None
    if _HAS_THOP:
        try:
            clip_model.visual.eval()
            dummy = torch.randn(1, 3, 224, 224, device=device).type(clip_model.dtype)
            macs, _ = _thop_profile(clip_model.visual, inputs=(dummy,), verbose=False)
        except Exception:
            macs = None
    if macs is not None:
        print(f"[资源]总参数量={_fmt_m(total_params)}(+适配器={_fmt_m(adp_total)})|可训练={_fmt_m(adp_trainable)}|视觉MACs={_fmt_g(macs)} | 近似FLOPs={_fmt_g(2*macs)}/张@224")
    else:
        print(f"[资源]总参数量={_fmt_m(total_params)}(+适配器={_fmt_m(adp_total)})|可训练={_fmt_m(adp_trainable)}|视觉MACs=NA | 近似FLOPs=NA/张@224")

class GatedAdapter(nn.Module):
    def __init__(self, d_model, reduction=2, dropout=0.1,
                 init_scale=2e-3, gate_bias_init=-2.0):
        super().__init__()
        hidden_dim = d_model // reduction
        self.up = nn.Linear(d_model, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.down = nn.Linear(hidden_dim, d_model)
        self.gate = nn.Linear(d_model, 1)
        self.scale = nn.Parameter(torch.ones(1))

        with torch.no_grad():
            nn.init.xavier_uniform_(self.up.weight)
            nn.init.xavier_uniform_(self.down.weight)
            nn.init.zeros_(self.up.bias)
            nn.init.zeros_(self.down.bias)
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, gate_bias_init)

    def forward(self, x):
        h = self.up(x)
        h = self.act(h)
        h = self.dropout(h)
        h = self.down(h)
        g = self.sigmoid(self.gate(x))
        return self.scale * g * h


class LIAdapter(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256):
        super(LIAdapter, self).__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=4, batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.adapter = GatedAdapter(d_model=input_dim, reduction=2, dropout=0.1)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.output_layer = nn.Linear(input_dim, num_classes)

    def forward(self, img_feat, text_feat):
        img_feat = img_feat.unsqueeze(1)
        if text_feat.dim() == 2:
            text_feat = text_feat.unsqueeze(1)
        elif text_feat.dim() == 3 and text_feat.size(0) == 1:
            text_feat = text_feat.expand(img_feat.size(0), -1, -1)

        img_feat = img_feat.float()
        text_feat = text_feat.float()

        attn_output, _ = self.cross_attn(query=text_feat, key=img_feat, value=img_feat)
        x = self.norm1(text_feat + attn_output)

        ff_output = self.feed_forward(x)
        adp_output = self.adapter(x)
        x = self.norm2(x + ff_output + adp_output)

        x = x.squeeze(1)
        logits = self.output_layer(x)
        return logits


@torch.no_grad()
def _infer_probs(model_li: nn.Module, prompt_adapter: nn.Module, loader):
    model_li.eval()
    prompt_adapter.eval()
    preds = []
    for img_feat, labels, names in loader:
        img_feat = img_feat.to(device)
        text_feat = prompt_adapter(img_feat)
        outputs = torch.sigmoid(model_li(img_feat, text_feat)).cpu().numpy()
        for i, name in enumerate(names):
            row = {"Image_Name": name}
            for j, cname in enumerate(class_list):
                row[cname] = float(outputs[i][j])
            preds.append(row)
    return preds

def train_and_evaluate_with_val(train_data, val_data, test_data, lr, ratio, patience=5,
                                pretrained_path="li_adapter_coco_pretrained.pth"):

    base_train = train_data
    if FEWSHOT:
        base_train = build_fewshot_subset(train_data, class_list, shots=SHOTS, seed=seed)
    if ratio >= 1.0:
        cur_train = base_train
    else:
        n_train = max(1, int(len(base_train) * ratio))
        cur_train = random.sample(base_train, n_train)


    train_loader = _build_loader(cur_train, batch_size=32, shuffle=True)
    val_loader   = _build_loader(val_data,   batch_size=32, shuffle=False)
    test_loader  = _build_loader(test_data,  batch_size=32, shuffle=False)


    model_li = LIAdapter(512, num_classes).to(device)
    prompt_adapter = CoCoOpPromptAdapter(text_features,learnable_base=False).to(device)


    if pretrained_path and os.path.exists(pretrained_path):
        try:
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            cur_state = model_li.state_dict()
            matched = 0
            for name, param in pretrained_dict.items():
                if "output_layer" in name:
                    continue
                if name in cur_state and cur_state[name].shape == param.shape:
                    cur_state[name] = param
                    matched += 1
            model_li.load_state_dict(cur_state)
            print(f"载入预训练权重（匹配{matched}个非分类层参数）")
        except Exception as e:
            print(f"预训练权重加载失败：{e}")
    else:
        print("未使用预训练权重，随机初始化")


    pos_weight = compute_pos_weight(cur_train, class_list).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    gate_params = list(model_li.adapter.gate.parameters()) + [model_li.adapter.scale]
    adapter_core_params = [
        p for name, p in model_li.adapter.named_parameters()
        if ("gate" not in name and "scale" not in name)
    ]
    other_li_params = [
        p for name, p in model_li.named_parameters()
        if not name.startswith("adapter.")
    ]
    prompt_params = list(prompt_adapter.parameters())

    base_lr    = lr
    adapter_lr = lr * 0.5
    gate_lr    = lr * 0.1

    optimizer = optim.AdamW(
        [
            {"params": other_li_params,   "lr": base_lr},
            {"params": adapter_core_params,"lr": adapter_lr},
            {"params": gate_params,        "lr": gate_lr},
            {"params": prompt_params,      "lr": base_lr},
        ]
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=1, min_lr=1e-5
    )


    best_val_mAP = -1.0
    best_epoch = 0
    best_state = None
    best_prompt_state = None
    patience_counter = 0
    num_epochs = 20

    for epoch in range(num_epochs):
        model_li.train()
        prompt_adapter.train()
        total_loss = 0.0
        for img_feat, labels, _ in tqdm(
            train_loader, desc=f"比例={ratio:.2f} | 学习率={lr:.0e} | Epoch {epoch+1}/{num_epochs}"
        ):
            img_feat, labels = img_feat.to(device), labels.to(device)
            text_feat = prompt_adapter(img_feat)
            logits = model_li(img_feat, text_feat)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model_li.parameters()) + list(prompt_adapter.parameters()),
                max_norm=1.0
            )
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))
        print(f"Epoch {epoch+1}: 训练损失={avg_loss:.4f}")


        val_preds = _infer_probs(model_li, prompt_adapter, val_loader)
        val_mAP = calculate_mAP(val_data, val_preds)
        val_acc, val_prec, val_rec, val_f1 = compute_metrics(val_data, val_preds)
        print(f"Epoch {epoch+1}: 验证mAP={val_mAP:.4f}, 验证F1={val_f1:.4f}, 验证准确率={val_acc:.4f}")

        scheduler.step(val_mAP)
        print(f"当前学习率 = {optimizer.param_groups[0]['lr']:.6g}")


        if val_mAP > best_val_mAP:
            best_val_mAP = val_mAP
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model_li.state_dict().items()}
            best_prompt_state = {k: v.detach().cpu().clone() for k, v in prompt_adapter.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停：Epoch {epoch+1}（验证mAP连续{patience}轮无提升）")
                break


    if best_state is not None:
        model_li.load_state_dict(best_state)
    if best_prompt_state is not None:
        prompt_adapter.load_state_dict(best_prompt_state)


    val_probs_for_th = _infer_probs(model_li, prompt_adapter, val_loader)
    best_ths = tune_thresholds_from_val(val_data, val_probs_for_th, class_list)


    test_preds = _infer_probs(model_li, prompt_adapter, test_loader)
    test_mAP = calculate_mAP(test_data, test_preds)

    y_true = np.array([[int(x[c]) for c in class_list] for x in test_data], dtype=int)
    y_prob = np.array([[row[c] for c in class_list] for row in test_preds], dtype=float)
    y_hat  = (y_prob >= best_ths[None, :]).astype(int)

    test_acc  = accuracy_score(y_true.ravel(), y_hat.ravel())
    test_prec = precision_score(y_true.ravel(), y_hat.ravel(), zero_division=0)
    test_rec  = recall_score(y_true.ravel(), y_hat.ravel(), zero_division=0)
    test_f1   = f1_score(y_true.ravel(), y_hat.ravel(), zero_division=0)

    print(f"[最优验证mAP@Epoch {best_epoch}] 测试mAP={test_mAP:.4f}, 测试F1={test_f1:.4f}, 测试准确率={test_acc:.4f}")
    cnki = compute_cnki_metrics(y_true, y_hat)
    print(f"[测试-CNKI]{json.dumps(cnki, ensure_ascii=False)}")


    report_params_flops_adapter(clip_model, model_li)

    result = {
        "训练比例": ratio,
        "学习率": lr,
        "最优轮次": best_epoch,
        "验证mAP": best_val_mAP,
        "测试mAP": test_mAP,
        "测试准确率": test_acc,
        "测试精确率": test_prec,
        "测试召回率": test_rec,
        "测试F1": test_f1,
        "训练集大小": len(cur_train),
        "验证集大小": len(val_data),
        "测试集大小": len(test_data),
        "FewShot模式": FEWSHOT,
        "Shots数": SHOTS if FEWSHOT else 0,
        "最优阈值": best_ths.tolist(),
        "CNKI指标": cnki,
    }
    return result


def main(args):
    ratios = [1.0]
    learning_rates = [9e-3]
    all_results = []


    if args.all_shots:
        shots_list = [1, 2, 4, 8, 16]
    else:
        shots_list = [args.shots]

    print(f"训练集样本数: {len(train_all)}, 验证集样本数: {len(val_all)}, 测试集样本数: {len(test_all)}")

    for cur_shots in shots_list:
        global SHOTS
        SHOTS = cur_shots
        print(f"\n==============================")
        print(f"当前 SHOTS = {SHOTS} （Few-shot 模式: {FEWSHOT}）")
        print(f"==============================")

        for ratio in ratios:
            print(f"\n===== 当前训练比例: {ratio:.2f} =====")
            for lr in learning_rates:
                print(f"\n当前学习率: {lr}")
                metrics = train_and_evaluate_with_val(
                    train_all, val_all, test_all, lr, ratio, patience=5
                )
                all_results.append(metrics)


    df = pd.DataFrame(all_results)
    best_row = df.loc[df["验证mAP"].idxmax()]
    print("\n⭐ 最优配置：")
    print(best_row)
    os.makedirs("results2", exist_ok=True)
    json_path = "results2/result.json"
    csv_path  = "results2/result.csv"
    df.to_json(json_path, orient="records", indent=2, force_ascii=False)
    df.to_csv(csv_path, index=False)
    print(f"\n所有结果已保存至：\n{json_path}\n{csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default=SHOTS, help="few-shot模式下每类样本数")
    parser.add_argument("--all_shots", action="store_true", help="运行shots=1,2,4,8,16")
    args = parser.parse_args()
    main(args)