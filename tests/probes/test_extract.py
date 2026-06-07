import numpy as np
from scripts.probes.extract import load_model, build_prompt, extract_resid

MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def test_extract_shape_and_determinism():
    model, tok = load_model(MODEL, device="cpu")
    msgs = [{"role": "user", "content": "hello"}]
    prompts = [build_prompt(msgs, None, tok), build_prompt(msgs, None, tok)]
    layers = [0, 1]
    a = extract_resid(model, tok, prompts, layers, batch_size=2)
    b = extract_resid(model, tok, prompts, layers, batch_size=1)
    assert a.shape[0] == 2 and a.shape[1] == 2          # [N, n_layers, D]
    assert a.shape[2] == model.config.hidden_size
    assert np.allclose(a, b, atol=1e-4)                 # batch-size invariant
