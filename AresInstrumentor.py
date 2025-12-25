"""
ares_instrumentation.py
Instrumentor adapté pour Ares (MoE/Transformer) et ETO (Cognitive).

Usage dans ton script d'entraînement Ares :
    from ares_instrumentation import AresInstrumentor
    
    # Instanciation
    instr = AresInstrumentor(model, optimizer, log_dir="runs/ares_v2", sample_rate=50)
    
    # Enregistrement intelligent (Ares spécifique)
    instr.register_moe_gates(model) # Trouve auto les gates MoE
    instr.register_attention_layers(model) # Trouve les layers d'attention

    # Boucle
    instr.on_epoch_start(epoch)
    ...
    loss.backward()
    instr.on_backward_complete() # Capture les gradients
    ...
    instr.on_batch_end(idx, current_tokens_generated)
"""

import os
import time
import math
import json
import re
from collections import defaultdict, deque
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# UTILITAIRES SPÉCIAUX ARES
# -------------------------
def calculate_perplexity(loss_value):
    """Calcul la perplexité safe (évite overflow)."""
    try:
        return math.exp(loss_value)
    except OverflowError:
        return float('inf')

def safe_mean(x):
    x = [v for v in x if v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))]
    return float(np.mean(x)) if x else 0.0

def tensor_stats(t: torch.Tensor) -> Dict[str, float]:
    if t is None: return {}
    # Pour Ares, on évite de copier de trop gros tenseurs vers le CPU si pas nécessaire
    if t.numel() > 1_000_000: # Si très gros tenseur, on sample
        # Échantillonnage aléatoire pour la vitesse
        indices = torch.randint(0, t.numel(), (10000,), device=t.device)
        a = t.view(-1)[indices].detach().cpu().float().numpy()
    else:
        a = t.detach().cpu().float().view(-1).numpy()
        
    if a.size == 0: return {"mean": 0.0, "std": 0.0}
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "max": float(a.max()),
        "abs_mean": float(np.abs(a).mean())
    }

# -------------------------
# COLLECTEURS SPÉCIALISÉS
# -------------------------

class MoEGateCollector:
    """
    Spécifique pour Ares : Surveille l'activité des 'Gates' (routeurs) des Experts.
    Permet de détecter le 'Expert Collapse' (quand un seul expert bosse).
    """
    def __init__(self):
        self.hooks = []
        self.expert_usage = defaultdict(lambda: deque(maxlen=100)) # layer -> history of entropy/usage

    def _make_hook(self, name):
        def hook(module, inp, out):
            # Dans un MoE, la sortie du Gating est souvent des probabilités [Batch, Seq, Num_Experts]
            # ou [Batch, Num_Experts]
            try:
                gates = out[0] if isinstance(out, tuple) else out
                if isinstance(gates, torch.Tensor):
                    # On veut savoir si les gates sont variées.
                    # Calcul de l'entropie sur la dimension des experts (dernière dim)
                    probs = torch.softmax(gates.detach().float(), dim=-1)
                    avg_usage = probs.mean(dim=0).cpu().numpy() # Utilisation moyenne par expert dans le batch
                    
                    # Stocker l'écart type de l'utilisation (si 0 = distribution parfaite, si haut = déséquilibre)
                    balance_metric = float(np.std(avg_usage))
                    self.expert_usage[name].append(balance_metric)
            except Exception:
                pass
        return hook

    def register(self, module: nn.Module, name: str):
        self.hooks.append(module.register_forward_hook(self._make_hook(name)))

    def remove_all(self):
        for h in self.hooks: h.remove()
        self.hooks = []

class ActivationCollector:
    def __init__(self):
        self.hooks = []
        self.stats = defaultdict(lambda: deque(maxlen=200))

    def _make_hook(self, name):
        def hook(module, inp, out):
            try:
                t = out[0] if isinstance(out, (tuple, list)) else out
                if isinstance(t, torch.Tensor):
                    s = tensor_stats(t)
                    self.stats[name].append(s)
            except Exception: pass
        return hook

    def register(self, module: nn.Module, name: str):
        self.hooks.append(module.register_forward_hook(self._make_hook(name)))

    def remove_all(self):
        for h in self.hooks: h.remove()

class GradientCollector:
    def __init__(self, model: nn.Module):
        self.model = model
        self.layer_grad_norms = defaultdict(list)

    def collect(self):
        total_norm = 0.0
        for name, p in self.model.named_parameters():
            if p.grad is not None:
                grad_norm = p.grad.detach().data.norm(2).item()
                self.layer_grad_norms[name].append(grad_norm)
                total_norm += grad_norm ** 2
        return total_norm ** 0.5 # Global norm

    def reset_epoch(self):
        self.layer_grad_norms = defaultdict(list)

# -------------------------
# VISUALISATEUR
# -------------------------
class TrainingVisualizer:
    def __init__(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir

    def plot_epoch_metrics(self, epoch: int, metrics: Dict[str, Any]):
        keys_to_plot = ["loss", "perplexity", "accuracy", "lr", "grad_norm_mean", "moe_imbalance"]
        present = [k for k in keys_to_plot if k in metrics and metrics[k] and len(metrics[k]) > 0]
        n = len(present)
        if n == 0: return None
        
        plt.figure(figsize=(10, 3*n))
        for i, k in enumerate(present, start=1):
            plt.subplot(n, 1, i)
            data = metrics[k]
            plt.plot(data, label=k, color='tab:blue')
            # Lissage pour les courbes bruyantes (Loss/PPL)
            if len(data) > 20 and k in ["loss", "perplexity"]:
                kernel_size = 10
                kernel = np.ones(kernel_size) / kernel_size
                smoothed = np.convolve(data, kernel, mode='valid')
                plt.plot(range(len(data)-len(smoothed), len(data)), smoothed, color='tab:orange', linestyle='--', label='Smoothed')
            
            plt.title(f"Ares Training - {k.upper()}")
            plt.grid(True, alpha=0.3)
            plt.legend()
            
        fname = os.path.join(self.out_dir, f"epoch_{epoch:04d}_stats.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=120)
        plt.close()
        return fname

# -------------------------
# INSTRUMENTOR ARES
# -------------------------
class AresInstrumentor:
    """
    Instrumentor central pour Ares.
    """
    def __init__(self, model: nn.Module, optimizer: Optional[torch.optim.Optimizer]=None, *,
                 log_dir: str = "runs/ares_experiment", tb_enabled: bool = True,
                 sample_rate: int = 50,
                 device: Optional[torch.device] = None):
        
        self.model = model
        self.optimizer = optimizer
        self.sample_rate = sample_rate
        self.device = device or (next(model.parameters()).device if list(model.parameters()) else torch.device("cpu"))
        
        # Collectors
        self.activation_collector = ActivationCollector()
        self.moe_collector = MoEGateCollector() # SPÉCIFIQUE ARES
        self.grad_collector = GradientCollector(model)
        
        # Logging
        self.vis = TrainingVisualizer(out_dir=os.path.join(log_dir, "figs"))
        self.tb_enabled = tb_enabled
        self.writer = SummaryWriter(log_dir) if tb_enabled else None
        self.log_dir = log_dir

        # State
        self.current = {
            "epoch": 0, "batch": 0,
            "losses": [], "ppls": [], "accs": [], "times": [],
            "moe_balance": []
        }
        self.history = defaultdict(list)
        self.start_time = time.time()

        print(f"✅ [AresInstrumentor] Initialisé sur {self.log_dir}")

    # --- AUTO REGISTRATION INTELLIGENTE ---
    def register_moe_gates(self, model: nn.Module):
        """Cherche automatiquement les couches qui ressemblent à des Gates MoE."""
        count = 0
        for name, module in model.named_modules():
            # Détection heuristique basée sur les noms communs dans Ares
            if "gate" in name.lower() or "router" in name.lower():
                self.moe_collector.register(module, name)
                count += 1
        print(f"🔧 [AresInstrumentor] {count} MoE Gates enregistrées pour monitoring.")

    def register_layers(self, layer_names_substrings: List[str] = ["attn", "mlp"]):
        """Enregistre les activations pour les couches contenant certaines substrings."""
        count = 0
        for name, module in self.model.named_modules():
            # On évite d'enregistrer tout le modèle, juste les blocks clés
            if any(s in name for s in layer_names_substrings) and not isinstance(module, (nn.ModuleList, nn.Sequential)):
                if isinstance(module, (nn.Linear, nn.LayerNorm)):
                    self.activation_collector.register(module, name)
                    count += 1
        print(f"🔧 [AresInstrumentor] {count} Layers enregistrés (Attention/MLP) pour monitoring stats.")

    # --- BOUCLE EVENTS ---
    def on_epoch_start(self, epoch: int):
        self.current["epoch"] = epoch
        self.current["batch"] = 0
        self.current["losses"] = []
        self.current["ppls"] = [] # Reset Perplexity
        self.current["accs"] = []
        self.grad_collector.reset_epoch()
        print(f"\n--- 🚀 EPOCH {epoch} START ---")

    def on_batch_start(self, batch_idx: int):
        self.current["batch"] = batch_idx
        self._batch_t0 = time.time()

    def on_loss_computed(self, loss: torch.Tensor, outputs=None, targets=None):
        """
        Calcul loss et PPL (Perplexity).
        """
        loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
        ppl_val = calculate_perplexity(loss_val)
        
        self.current["losses"].append(loss_val)
        self.current["ppls"].append(ppl_val)

        # Calcul Accuracy Rapide (Top-1)
        if outputs is not None and targets is not None:
            with torch.no_grad():
                # Hypothèse shape outputs: [Batch, Seq, Vocab] ou [Batch*Seq, Vocab]
                if outputs.dim() == 3:
                    pred = outputs.argmax(dim=-1).view(-1)
                    tgt = targets.view(-1)
                else:
                    pred = outputs.argmax(dim=-1)
                    tgt = targets
                
                # Ignorer padding si possible (si target == -100 ou 0)
                mask = (tgt != -100) # Standard PyTorch ignore_index
                if mask.sum() > 0:
                    acc = (pred[mask] == tgt[mask]).float().mean().item()
                    self.current["accs"].append(acc)

    def on_backward_complete(self):
        """Appeler juste après loss.backward() et avant optimizer.step()"""
        global_norm = self.grad_collector.collect()
        
        # Alerte Gradient Explosion
        if global_norm > 5.0 and self.current["batch"] % 10 == 0:
            print(f"⚠️ [Warning] Gradient Norm High: {global_norm:.2f}")
        
        if self.tb_enabled and self.writer:
            step = self.current["epoch"] * 10000 + self.current["batch"] # Approx step
            self.writer.add_scalar("grad/global_norm", global_norm, step)

    def on_batch_end(self, batch_idx: int):
        dt = time.time() - self._batch_t0
        self.current["times"].append(dt)

        if batch_idx % self.sample_rate == 0:
            avg_loss = safe_mean(self.current["losses"][-self.sample_rate:])
            avg_ppl = safe_mean(self.current["ppls"][-self.sample_rate:])
            avg_acc = safe_mean(self.current["accs"][-self.sample_rate:])
            
            # Check MoE balance
            moe_imbalance = 0.0
            for name, deque_vals in self.moe_collector.expert_usage.items():
                if deque_vals: moe_imbalance += deque_vals[-1]
            moe_msg = f" | MoE Imb: {moe_imbalance:.3f}" if moe_imbalance > 0 else ""

            print(f"Batch {batch_idx:04d} | Loss: {avg_loss:.4f} | PPL: {avg_ppl:.2f} | Acc: {avg_acc:.1%} | {dt*1000:.0f}ms{moe_msg}")

            if self.writer:
                step = self.current["epoch"] * 10000 + batch_idx # Hack pour step continu
                self.writer.add_scalar("train/loss", avg_loss, step)
                self.writer.add_scalar("train/perplexity", avg_ppl, step)
                self.writer.add_scalar("train/accuracy", avg_acc, step)

    def on_epoch_end(self, epoch: int):
        # Sauvegarde History
        self.history["loss"].extend(self.current["losses"])
        self.history["perplexity"].extend(self.current["ppls"])
        self.history["accuracy"].extend(self.current["accs"])
        
        avg_loss = safe_mean(self.current["losses"])
        avg_ppl = safe_mean(self.current["ppls"])
        
        print(f"🏁 EPOCH {epoch} DONE | Avg Loss: {avg_loss:.4f} | Avg PPL: {avg_ppl:.2f}")

        # Visualisation
        try:
            metrics_dict = {
                "loss": self.history["loss"],
                "perplexity": self.history["perplexity"],
                "accuracy": self.history["accuracy"],
                "grad_norm_mean": [np.mean(vals) for vals in self.grad_collector.layer_grad_norms.values()] if self.grad_collector.layer_grad_norms else []
            }
            fname = self.vis.plot_epoch_metrics(epoch, metrics_dict)
            if fname: print(f"📊 Graphique sauvegardé : {fname}")
        except Exception as e:
            print(f"Erreur plotting: {e}")

        # Tensorboard Weights Histogram (Lourd, on le fait 1 fois par epoch)
        if self.writer:
            for name, param in self.model.named_parameters():
                if "weight" in name and param.requires_grad:
                    self.writer.add_histogram(f"weights/{name}", param, epoch)

    def log_text_sample(self, text_input: str, text_output: str, step: int):
        """Pour logger ce que Ares raconte dans Tensorboard."""
        if self.writer:
            self.writer.add_text("generation", f"**In:** {text_input}\n\n**Ares:** {text_output}", step)

    def close(self):
        if self.writer: self.writer.close()
        self.activation_collector.remove_all()
        self.moe_collector.remove_all()
        print("[AresInstrumentor] Fermeture et nettoyage.")