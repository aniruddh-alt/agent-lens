"""Load a model and pull residual-stream activations at the LAST PROMPT TOKEN.
Reading prompt tokens (not generation) makes the probe robust to the
premature-EOS artifact: the activations exist whether or not the model emits."""
from __future__ import annotations
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(path: str, device: str | None = None):
    if device is None:  # auto-detect so the same code runs on the pod (cuda) and locally (cpu)
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"  # so last-real-token index = attn_mask.sum()-1
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype).to(device)
    model.eval()
    return model, tok


def build_prompt(messages, tools, tok) -> str:
    return tok.apply_chat_template(messages, tools=tools,
                                   add_generation_prompt=True, tokenize=False)


def assert_tool_template_differs(tok, messages, tools):
    """Abort if the tokenizer's chat template does not actually branch on `tools`
    (then the tool- and chat-format prompts are byte-identical and the tool/chat
    distinction -- including Phase B1's format decomposition -- is a silent no-op).
    The local tiny-test tokenizer fails this; real Llama-3.1-8B-Instruct passes it."""
    if build_prompt(messages, tools, tok) == build_prompt(messages, None, tok):
        raise SystemExit(
            "FATAL: tool-format prompt == chat-format prompt for this tokenizer -- "
            "its chat_template ignores `tools`, so the tool/chat distinction (and "
            "Phase B1) would be a silent no-op. Copy the base Llama-3.1 tokenizer "
            "onto this checkpoint (plan Task 0.3) before probing.")


@torch.no_grad()
def extract_resid(model, tok, prompts, layers, batch_size: int = 8, max_length: int = 4096):
    """Returns float32 np.ndarray [n_prompts, n_layers, d_model]."""
    cache, chunks = {}, {L: [] for L in layers}
    def mk_hook(L):
        def hook(_m, _i, out):
            cache[L] = out[0] if isinstance(out, tuple) else out
        return hook
    handles = [model.model.layers[L].register_forward_hook(mk_hook(L)) for L in layers]
    try:
        for i in range(0, len(prompts), batch_size):
            enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                      truncation=True, max_length=max_length).to(model.device)
            model(**enc)
            last = enc.attention_mask.sum(dim=1) - 1            # [B] index of last real token
            ar = torch.arange(enc.input_ids.size(0), device=model.device)
            for L in layers:
                h = cache[L]                                    # [B, T, D]
                chunks[L].append(h[ar, last].float().cpu().numpy())
    finally:
        for hh in handles:
            hh.remove()
    return np.stack([np.concatenate(chunks[L], 0) for L in layers], axis=1)
