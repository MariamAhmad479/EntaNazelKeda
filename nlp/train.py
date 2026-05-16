"""
train.py  --  Two-phase DistilBERT fine-tuning for WardrobeNLPModel.

Phase 1 (epochs 1-3)  : bottom 4 encoder layers frozen, lr=2e-5
Phase 2 (epochs 4-5)  : all layers unfrozen, lr=5e-6

Saves best checkpoint (lowest val loss) to nlp/saved_models_bert/.
The old BiLSTM weights in nlp/saved_models/ are untouched.
"""

import json, os, sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import f1_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp.dataset import (
    get_bert_dataloaders,
    OCCASIONS, WEATHER_CLASSES, STYLES, INTENT_CLASSES,
    OCCASION_UNKNOWN_IDX, WEATHER_UNKNOWN_IDX, STYLE_UNKNOWN_IDX, INTENT_UNKNOWN_IDX,
)
from nlp.model import DistilBertMultiTaskClassifier

# ── Config ────────────────────────────────────────────────────────────────────
PHASE1_EPOCHS   = 3
PHASE2_EPOCHS   = 2
BATCH_SIZE      = 32
PHASE1_LR       = 2e-5
PHASE2_LR       = 5e-6
FREEZE_N_LAYERS = 4
NUM_SYNTHETIC   = 10000
MAX_LEN         = 64
LOSS_WEIGHTS    = {"intent": 1.0, "occasion": 1.5, "weather": 1.0, "style": 1.0}
SAVE_DIR        = os.path.join(os.path.dirname(__file__), "saved_models_bert")
INTENT_CONF_THRESH   = 0.60

OCCASION_CONF_THRESH = 0.25
WEATHER_CONF_THRESH  = 0.35
STYLE_CONF_THRESH    = 0.30


# ── Helpers ───────────────────────────────────────────────────────────────────

def masked_correct(logits, labels):
    pred = logits.argmax(dim=1)
    return (pred == labels).sum().item(), labels.size(0)

def collect_preds(logits, labels):
    return logits.argmax(dim=1).cpu().tolist(), labels.cpu().tolist()


def compute_coverage(logit_batches, threshold):
    import torch.nn.functional as F
    all_logits = torch.cat(logit_batches, dim=0)
    probs = F.softmax(all_logits, dim=1).max(dim=1).values
    return (probs >= threshold).float().mean().item()


def print_cm(cm, names, title):
    pad = max(max(len(n) for n in names), 6)
    print(f"\n  Confusion Matrix -- {title}:")
    print(" " * (pad + 2) + "  ".join(f"{n:>{pad}}" for n in names))
    for i, row_name in enumerate(names):
        print(f"  {row_name:>{pad}} " + "  ".join(f"{cm[i,j]:>{pad}}" for j in range(len(names))))
    print()


def run_epoch(model, loader, device, criterions, optimizer=None, scheduler=None):
    """One train or eval pass. Returns (avg_loss, acc_occ, acc_wea, acc_sty, preds_dict)."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    int_c = int_n = occ_c = occ_n = wea_c = wea_n = sty_c = sty_n = 0
    all_int_p, all_int_l = [], []
    all_occ_p, all_occ_l = [], []
    all_wea_p, all_wea_l = [], []
    all_sty_p, all_sty_l = [], []
    int_buf, occ_buf, wea_buf, sty_buf = [], [], [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            i_lb = batch["intent_label"].to(device)
            o_lb = batch["occasion_label"].to(device)
            w_lb = batch["weather_label"].to(device)
            s_lb = batch["style_label"].to(device)

            i_lg, o_lg, w_lg, s_lg = model(ids, mask)

            loss = (
                LOSS_WEIGHTS["intent"]   * criterions[0](i_lg, i_lb)
              + LOSS_WEIGHTS["occasion"] * criterions[1](o_lg, o_lb)
              + LOSS_WEIGHTS["weather"]  * criterions[2](w_lg, w_lb)
              + LOSS_WEIGHTS["style"]    * criterions[3](s_lg, s_lb)
            )

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler:
                    scheduler.step()

            total_loss += loss.item()
            c, n = masked_correct(i_lg, i_lb);   int_c += c; int_n += n
            c, n = masked_correct(o_lg, o_lb); occ_c += c; occ_n += n
            c, n = masked_correct(w_lg, w_lb);  wea_c += c; wea_n += n
            c, n = masked_correct(s_lg, s_lb);    sty_c += c; sty_n += n
            p, l = collect_preds(i_lg, i_lb);    all_int_p += p; all_int_l += l
            p, l = collect_preds(o_lg, o_lb);  all_occ_p += p; all_occ_l += l
            p, l = collect_preds(w_lg, w_lb);   all_wea_p += p; all_wea_l += l
            p, l = collect_preds(s_lg, s_lb);     all_sty_p += p; all_sty_l += l
            int_buf.append(i_lg.detach().cpu())
            occ_buf.append(o_lg.detach().cpu())
            wea_buf.append(w_lg.detach().cpu())
            sty_buf.append(s_lg.detach().cpu())

    n_batches = len(loader)
    int_f1 = f1_score(all_int_l, all_int_p, average="macro", zero_division=0) if all_int_l else 0.0
    occ_f1 = f1_score(all_occ_l, all_occ_p, average="macro", zero_division=0) if all_occ_l else 0.0
    wea_f1 = f1_score(all_wea_l, all_wea_p, average="macro", zero_division=0) if all_wea_l else 0.0
    sty_f1 = f1_score(all_sty_l, all_sty_p, average="macro", zero_division=0) if all_sty_l else 0.0

    return {
        "loss": total_loss / n_batches,
        "acc": (int_c/int_n if int_n else 0, occ_c/occ_n if occ_n else 0, wea_c/wea_n if wea_n else 0, sty_c/sty_n if sty_n else 0),
        "f1":  (int_f1, occ_f1, wea_f1, sty_f1),
        "preds": (all_int_p, all_int_l, all_occ_p, all_occ_l, all_wea_p, all_wea_l, all_sty_p, all_sty_l),
        "logit_bufs": (int_buf, occ_buf, wea_buf, sty_buf),
    }


# ── Main training function ────────────────────────────────────────────────────

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Building DataLoaders (downloading DistilBERT tokenizer if needed)...")
    train_loader, val_loader, tokenizer = get_bert_dataloaders(
        num_synthetic=NUM_SYNTHETIC, batch_size=BATCH_SIZE, max_len=MAX_LEN,
    )
    print(f"Train: {len(train_loader.dataset):,} samples | Val: {len(val_loader.dataset):,} samples")

    model = DistilBertMultiTaskClassifier(
        num_intents=len(INTENT_CLASSES) + 1,
        num_occasions=len(OCCASIONS) + 1,
        num_weather=len(WEATHER_CLASSES) + 1,
        num_styles=len(STYLES) + 1,
        dropout=0.3,
    ).to(device)

    criterions = (
        nn.CrossEntropyLoss(),
        nn.CrossEntropyLoss(),
        nn.CrossEntropyLoss(),
        nn.CrossEntropyLoss(),
    )

    best_val_loss = float("inf")
    best_state    = None
    total_epochs  = PHASE1_EPOCHS + PHASE2_EPOCHS

    for phase in (1, 2):
        epochs = PHASE1_EPOCHS if phase == 1 else PHASE2_EPOCHS
        lr     = PHASE1_LR     if phase == 1 else PHASE2_LR

        if phase == 1:
            print(f"\n=== Phase 1: freeze bottom {FREEZE_N_LAYERS} layers, lr={PHASE1_LR} ===")
            model.freeze_bottom_layers(FREEZE_N_LAYERS)
        else:
            print(f"\n=== Phase 2: unfreeze all layers, lr={PHASE2_LR} ===")
            model.unfreeze_all_encoder()

        print(f"Trainable params: {model.trainable_params():,}")

        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr, weight_decay=0.01,
        )
        scheduler = OneCycleLR(
            optimizer, max_lr=lr,
            steps_per_epoch=len(train_loader), epochs=epochs,
            pct_start=0.1,
        )

        for ep in range(1, epochs + 1):
            tr = run_epoch(model, train_loader, device, criterions, optimizer, scheduler)
            va = run_epoch(model, val_loader,   device, criterions)

            print(
                f"  Ep {ep}/{epochs}  "
                f"Loss {tr['loss']:.4f}/{va['loss']:.4f}  "
                f"Acc Int/Occ/Wea {va['acc'][0]:.2f}/{va['acc'][1]:.2f}/{va['acc'][2]:.2f}  "
                f"F1 {va['f1'][0]:.2f}/{va['f1'][1]:.2f}/{va['f1'][2]:.2f}"
            )

            if va["loss"] < best_val_loss:
                best_val_loss = va["loss"]
                best_state    = {k: v.clone() for k, v in model.state_dict().items()}
                print(f"    -> new best val loss: {best_val_loss:.4f}")

    # ── Final diagnostics ─────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    print("\n" + "="*65)
    print("TRAINING COMPLETE")
    print("="*65)

    va = run_epoch(model, val_loader, device, criterions)
    int_p, int_l, occ_p, occ_l, wea_p, wea_l, sty_p, sty_l = va["preds"]
    int_buf, occ_buf, wea_buf, sty_buf = va["logit_bufs"]

    if int_l:
        print_cm(confusion_matrix(int_l, int_p, labels=list(range(len(INTENT_CLASSES)))), INTENT_CLASSES, "Intent")
    if occ_l:
        print_cm(confusion_matrix(occ_l, occ_p, labels=list(range(len(OCCASIONS)))),    OCCASIONS,       "Occasion")
    if wea_l:
        print_cm(confusion_matrix(wea_l, wea_p, labels=list(range(len(WEATHER_CLASSES)))), WEATHER_CLASSES, "Weather")
    if sty_l:
        print_cm(confusion_matrix(sty_l, sty_p, labels=list(range(len(STYLES)))),       STYLES,          "Style")

    cov_int = compute_coverage(int_buf, INTENT_CONF_THRESH)
    cov_occ = compute_coverage(occ_buf, OCCASION_CONF_THRESH)
    cov_wea = compute_coverage(wea_buf, WEATHER_CONF_THRESH)
    cov_sty = compute_coverage(sty_buf, STYLE_CONF_THRESH)
    print(f"Coverage (above confidence threshold):")
    print(f"  Intent {cov_int:.1%} Occasion {cov_occ:.1%}  Weather {cov_wea:.1%}  Style {cov_sty:.1%}")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(best_state, os.path.join(SAVE_DIR, "nlp_model.pth"))
    tokenizer.save_pretrained(SAVE_DIR)

    config = {
        "num_intents":   len(INTENT_CLASSES) + 1,
        "num_occasions": len(OCCASIONS) + 1,
        "num_weather":   len(WEATHER_CLASSES) + 1,
        "num_styles":    len(STYLES) + 1,
        "dropout":       0.3,
        "max_len":       MAX_LEN,
        "backend":       "distilbert",
    }
    with open(os.path.join(SAVE_DIR, "model_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved best model (val loss={best_val_loss:.4f}) -> {SAVE_DIR}")


if __name__ == "__main__":
    train_model()
