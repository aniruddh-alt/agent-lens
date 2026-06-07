import numpy as np
import torch
from scripts.probes.steer import _add_hook, _ablate_hook


def test_ablate_removes_component_and_add_shifts_it():
    d = np.zeros(8, np.float32); d[0] = 1.0          # unit direction along axis 0
    d_t = torch.tensor(d)
    h = torch.tensor([[3.0, 1.0] + [0.0] * 6])       # projection onto d = 3.0

    # true ablation: project the component out -> projection ~ 0
    ablated = _ablate_hook(d_t)(None, None, h.clone())
    assert torch.allclose((ablated * d_t).sum(-1), torch.zeros(1), atol=1e-5)

    # additive steering: +alpha along d -> projection increases by alpha
    added = _add_hook(d_t, 5.0)(None, None, h.clone())
    assert torch.allclose((added * d_t).sum(-1), torch.tensor([8.0]), atol=1e-5)


def test_hooks_preserve_tuple_outputs():
    d_t = torch.zeros(4); d_t[0] = 1.0
    h = torch.ones((1, 1, 4))
    out = _add_hook(d_t, 1.0)(None, None, (h, "kv"))     # decoder layer may return a tuple
    assert isinstance(out, tuple) and out[1] == "kv"
