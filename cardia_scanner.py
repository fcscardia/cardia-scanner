import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextvars import ContextVar
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import roc_auc_score

# =============================================================================
# 1. ESTADO GLOBAL THREAD-SAFE (Isolado por Instância)
# =============================================================================
_interventions: ContextVar[Optional[Dict[Tuple[int, int], List[dict]]]] = ContextVar("_interventions", default=None)

# =============================================================================
# 2. ABSTRAÇÃO DE ARQUITETURA
# =============================================================================
@dataclass(frozen=True)
class HFDecoderAdapter:
    """Resolve discrepâncias de configuração entre Qwen, Gemma, Llama, etc."""
    @staticmethod
    def get_text_config(model: nn.Module):
        cfg = getattr(model, "config", None)
        if callable(getattr(cfg, "get_text_config", None)):
            return cfg.get_text_config()
        if cfg is None and hasattr(model, "model"):
            cfg = getattr(model.model, "config", None)
        if hasattr(cfg, "text_config"):
            cfg = cfg.text_config
        if cfg is None:
            raise ValueError("Não foi possível resolver o config do modelo.")
        return cfg

    @staticmethod
    def get_dimensions(model: nn.Module) -> Tuple[int, int]:
        cfg = HFDecoderAdapter.get_text_config(model)
        num_layers = next((getattr(cfg, k) for k in ["num_hidden_layers", "n_layer", "num_layers"] if hasattr(cfg, k)), None)
        hidden_size = next((getattr(cfg, k) for k in ["hidden_size", "n_embd", "d_model"] if hasattr(cfg, k)), None)
        
        if num_layers is None or hidden_size is None:
            raise AttributeError("Falha ao inferir num_layers ou hidden_size do config.")
        return int(num_layers), int(hidden_size)

# =============================================================================
# 3. SCANNER CARDIA (v0.1 - PoC Definitivo)
# =============================================================================
class CardiaTensorScanner:
    def __init__(self, model: nn.Module, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
        # Herda o tipo de dados nativo do modelo em vez de forçar bfloat16
        try:
            self.dtype = next(model.parameters()).dtype
        except StopIteration:
            self.dtype = torch.float32
        
        # ATENÇÃO: Altera o estado do tokenizer do usuário para permitir inferência em batch
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = getattr(self.tokenizer, "eos_token_id", 0)
        
        # Introspecção
        self.adapter = HFDecoderAdapter()
        self.num_layers, self.hidden_size = self.adapter.get_dimensions(model)
        self.num_states = self.num_layers + 1
        
        # Estado Interno
        self.directions: Dict[int, np.ndarray] = {}
        self.midpoints: Dict[int, float] = {}
        self._tensor_directions: Dict[int, torch.Tensor] = {}
        
        # Hooks Seguros
        self._handles = []
        self._install_persistent_hooks()

    def close(self):
        """Remove os hooks da instância atual (limpeza de recursos)."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # -------------------------------------------------------------------------
    # GESTÃO DE ÍNDICES E FORMATOS
    # -------------------------------------------------------------------------
    def _normalize_state_layer(self, layer: int) -> int:
        """Índice para ler hidden_states (0 a num_layers)."""
        l = int(layer)
        if not -self.num_states <= l < self.num_states:
            raise IndexError(f"Estado {l} fora dos limites [-{self.num_states}, {self.num_states - 1}].")
        return l % self.num_states

    def _format_prompts(self, texts: List[str]) -> Tuple[List[str], bool]:
        """Evita duplicar BOS/EOS formatando corretamente o template."""
        if getattr(self.tokenizer, "chat_template", None):
            prompts = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": t}], 
                    add_generation_prompt=True, tokenize=False
                ) for t in texts
            ]
            return prompts, False  # template já adiciona especiais
        return texts, True

    # -------------------------------------------------------------------------
    # EXTRAÇÃO OTIMIZADA
    # -------------------------------------------------------------------------
    @torch.inference_mode()
    def extract(self, texts: List[str], layers: List[int], batch_size: int = 16) -> Dict[int, np.ndarray]:
        self.model.eval()
        norm_layers = list(dict.fromkeys(self._normalize_state_layer(l) for l in layers))
        acts = {l: [] for l in norm_layers}
        
        # Device fallback adaptativo
        input_device = next(self.model.parameters()).device
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            prompts, add_special = self._format_prompts(batch_texts)
            
            inputs = self.tokenizer(
                prompts, padding=True, return_tensors="pt", 
                add_special_tokens=add_special
            ).to(input_device)
            
            with torch.autocast(device_type=input_device.type, dtype=self.dtype):
                out = self.model(**inputs, output_hidden_states=True, return_dict=True)
            
            for l in norm_layers:
                # Slicing imediato do último token: [:, -1, :] e conversão float32 para CPU
                acts[l].append(out.hidden_states[l][:, -1, :].float().cpu().numpy().copy())
            
            del out, inputs
            
        return {l: np.concatenate(v, axis=0) for l, v in acts.items()}

    # -------------------------------------------------------------------------
    # TREINAMENTO (Mass Mean)
    # -------------------------------------------------------------------------
    def fit_reference_direction(self, pos_texts: List[str], neg_texts: List[str], layers: List[int]):
        """
        Direção baseada na diferença de médias.
        Atenção: A polaridade (o que é positivo) depende estritamente da ordem 
        das listas passadas aqui (pos_texts - neg_texts).
        """
        pos = self.extract(pos_texts, layers)
        neg = self.extract(neg_texts, layers)
        
        for l in pos:
            mu_pos, mu_neg = pos[l].mean(0), neg[l].mean(0)
            v = mu_pos - mu_neg
            norm = np.linalg.norm(v)
            if norm < 1e-12:
                raise ValueError(f"Colapso direcional na camada {l}.")
            
            self.directions[l] = v / norm
            self.midpoints[l] = float(((mu_pos + mu_neg) / 2) @ self.directions[l])
        self._tensor_directions.clear()

    # -------------------------------------------------------------------------
    # SCORING E SELEÇÃO DE CAMADA
    # -------------------------------------------------------------------------
    def score(self, texts: List[str], layer: int) -> np.ndarray:
        """Retorna a projeção linear centralizada (distância ao midpoint)."""
        l = self._normalize_state_layer(layer)
        if l not in self.directions:
            raise KeyError(f"Geometria não treinada para a camada {l}.")
        acts = self.extract(texts, [l])[l]
        return (acts @ self.directions[l]) - self.midpoints[l]

    def scan_layers(self, texts: List[str], labels: List[int], layers: List[int]) -> Dict[int, float]:
        """Calcula o AUROC de cada camada informada. 1 = pos_texts, 0 = neg_texts."""
        labels = np.asarray(labels)
        if len(np.unique(labels)) < 2:
            raise ValueError("labels precisa conter as duas classes (0 e 1) para calcular AUROC.")
        
        aurocs = {}
        for l in layers:
            norm_l = self._normalize_state_layer(l)
            scores = self.score(texts, norm_l)
            aurocs[norm_l] = roc_auc_score(labels, scores)
        return aurocs

    def best_layer(self, val_texts: List[str], val_labels: List[int], layers: List[int]) -> int:
        """Escolhe a camada com o maior AUROC num conjunto de validação."""
        aurocs = self.scan_layers(val_texts, val_labels, layers)
        return max(aurocs, key=aurocs.get)

    @torch.inference_mode()
    def scan_prompt(self, text: str, layer: int = -1) -> Dict:
        """
        Extrai métricas geométricas e entropia ao longo de todos os tokens.
        """
        self.model.eval()
        l = self._normalize_state_layer(layer)
        prompt, add_special = self._format_prompts([text])
        input_device = next(self.model.parameters()).device
        
        inputs = self.tokenizer(prompt[0], return_tensors="pt", add_special_tokens=add_special).to(input_device)
        out = self.model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
        
        acts = out.hidden_states[l][0].float().cpu().numpy()
        logits = out.logits[0]
        
        # Entropia de Shannon (log_softmax nativo base 2 com chunking)
        chunk_size = 128
        entropies = []
        for i in range(0, logits.size(0), chunk_size):
            chunk_logits = logits[i : i + chunk_size].float()
            logp = F.log_softmax(chunk_logits, dim=-1)
            p = logp.exp()
            h = -torch.xlogy(p, p).sum(dim=-1) / np.log(2.0)
            entropies.append(h)
            
        entropies_np = torch.cat(entropies).cpu().numpy()
        
        deltas = np.zeros(len(acts))
        if len(acts) > 1:
            deltas[1:] = np.linalg.norm(acts[1:] - acts[:-1], axis=1)
            
        if l in self.directions:
            projs = (acts @ self.directions[l]) - self.midpoints.get(l, 0.0)
        else:
            projs = [None] * len(acts)
            
        tokens = [self.tokenizer.convert_ids_to_tokens(tid) for tid in inputs.input_ids[0].tolist()]
        
        return {
            "camada": l, 
            "trajectory": [
                {"token": t, "risco_projetado": float(p) if p is not None else None, "entropia_bits": float(e), "velocidade": float(v)}
                for t, p, e, v in zip(tokens, projs, entropies_np, deltas)
            ]
        }

    # -------------------------------------------------------------------------
    # INTERVENÇÕES ODA-ID (HOOKS ALINHADOS)
    # -------------------------------------------------------------------------
    def _find_layers(self) -> nn.ModuleList:
        candidates = ["model.layers", "model.language_model.layers", "transformer.h"]
        for path in candidates:
            obj = self.model
            try:
                for p in path.split("."): obj = getattr(obj, p)
                if isinstance(obj, nn.ModuleList) and len(obj) == self.num_layers: return obj
            except AttributeError:
                continue
        raise AttributeError(
            "Hierarquia de camadas não reconhecida. "
            "Atualize os candidates para suportar esta arquitetura específica."
        )

    def _install_persistent_hooks(self):
        sid = id(self)
        for idx, layer_module in enumerate(self._find_layers()):
            def make_hook(block_idx):
                # O bloco de índice 'block_idx' (0 a L-1) produz o estado 'block_idx + 1' (1 a L)
                hs_idx = block_idx + 1
                def hook_fn(module, args, output):
                    state = _interventions.get()
                    # O hook procura intervenções agendadas exatamente para o seu hs_idx correspondente
                    ops = state.get((sid, hs_idx)) if state else None
                    if not ops:
                        return output
                    
                    is_tuple = isinstance(output, tuple)
                    h = output[0] if is_tuple else output
                    orig_dtype = h.dtype
                    h = h.float() # Cálculos de intervenção em fp32
                    
                    for op in ops:
                        # Adapta dinamicamente ao device da ativação
                        v = op["tensor"].to(device=h.device, dtype=torch.float32)
                        if op["type"] == "steer":
                            h = h + (op["alpha"] * v)
                        elif op["type"] == "ablate":
                            v_norm = v / (torch.norm(v) + 1e-8)
                            proj = torch.matmul(h, v_norm).unsqueeze(-1)
                            h = h - (proj * v_norm)
                            
                    h = h.to(orig_dtype)
                    return (h,) + output[1:] if is_tuple else h
                return hook_fn
            self._handles.append(layer_module.register_forward_hook(make_hook(idx)))

    def _get_tensor(self, layer: int) -> torch.Tensor:
        if layer not in self._tensor_directions:
            v = torch.tensor(self.directions[layer], dtype=torch.float32)
            self._tensor_directions[layer] = v / (v.norm() + 1e-9)
        return self._tensor_directions[layer]

    @contextmanager
    def _intervene(self, layer: int, op: dict):
        l = self._normalize_state_layer(layer)
        if l == 0:
            raise ValueError("Não é possível intervir nos embeddings (camada 0) com o módulo de intervenção atual.")
        if l == self.num_states - 1:
            warnings.warn("Aviso: A direção da última camada costuma refletir o estado pós-norm, mas o hook atua pré-norm.")
            
        key = (id(self), l)
        op = {**op, "tensor": self._get_tensor(l)}
        
        current_state = dict(_interventions.get() or {})
        current_state[key] = current_state.get(key, []) + [op]
        token = _interventions.set(current_state)
        try:
            yield
        finally:
            _interventions.reset(token)

    def steer(self, layer: int, alpha: float):
        """Aplica um viés direcional (steering) no residual da camada informada."""
        return self._intervene(layer, {"type": "steer", "alpha": alpha})

    def ablate(self, layer: int):
        """Subtrai ortogonalmente o conceito aprendido do residual da camada informada."""
        return self._intervene(layer, {"type": "ablate"})

    # -------------------------------------------------------------------------
    # PERSISTÊNCIA
    # -------------------------------------------------------------------------
    def save_directions(self, filepath: str):
        if not filepath.endswith(".npz"): filepath += ".npz"
        layers = sorted(self.directions.keys())
        np.savez(
            filepath,
            layers=np.array(layers),
            directions=np.stack([self.directions[l] for l in layers]),
            midpoints=np.array([self.midpoints[l] for l in layers]),
            hidden_size=np.array(self.hidden_size),
            model_class=np.array(self.model.__class__.__name__)
        )

    def load_directions(self, filepath: str):
        if not filepath.endswith(".npz"): filepath += ".npz"
        with np.load(filepath, allow_pickle=False) as data:
            if "hidden_size" in data and int(data["hidden_size"]) != self.hidden_size:
                raise ValueError(f"Incompatível! Arquivo: dim {data['hidden_size']} | Modelo: dim {self.hidden_size}")
            
            self.directions = {int(l): d for l, d in zip(data["layers"], data["directions"])}
            self.midpoints = {int(l): float(m) for l, m in zip(data["layers"], data["midpoints"])}
            self._tensor_directions.clear()
