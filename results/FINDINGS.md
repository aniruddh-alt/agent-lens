# Findings

Plain-language summary of what Agent-Lens found. Technical JSON outputs live in the subdirectories below.

---

## The question

When you teach a safe model to use tools on benign tasks, why does it start complying with harmful ones?

We tested two stories:

- **Story A (deletion):** SFT removes the model's ability to recognize harm.
- **Story B (competition):** SFT keeps harm recognition but makes "be helpful and complete the task" win over "refuse."

---

## Finding 1 — Harm detection survives fine-tuning

We trained linear probes to read "is this request harmful?" from internal activations at the last prompt token (layer 12).

| Model | Held-out AUC | Transfer from base direction |
|---|---|---|
| Base | 0.95 | — |
| Agentic FFT | 0.93 | 0.96 |
| Safety mix FFT | 0.94 | 0.97 |
| Chat control FFT | 0.94 | 0.99 |

**Takeaway:** All three fine-tunes keep harm detection at ~94% AUC. The model still *knows* the request is bad. Story A is wrong for this setting.

Raw data: `linear-probes/*_tool_summary.json`, `linear-probes/transfer_tool.json`

---

## Finding 2 — Tool format alone collapses refusal (even before SFT)

On 40 matched EHR harmful requests, forced generation:

| Format | Base model refusal |
|---|---|
| Chat (plain text) | **95%** (38/40) |
| Tool (function call) | **0%** (0/40) |

The base model executes harmful tool calls directly — e.g. prescribing lethal doses, removing allergy flags — without refusing.

**Takeaway:** A large part of "agentic misalignment" is already present in the base instruct model when you switch from chat to tool output format. Safety training in chat does not automatically transfer to tool calls.

---

## Finding 3 — Steering the format direction is causal

We built a direction `d_format` = (activations on harmful-in-chat) − (activations on harmful-in-tool). Adding this direction back during tool-format generation restores refusal:

| Model | Refusal at c=0 | Refusal at c=1× natural gap |
|---|---|---|
| Base | 0% | **42%** |
| Agentic FFT | 0% | 15% |

Harmless prompts stay at 0% refusal across doses — the effect is specific to harmful requests.

**Takeaway:** The chat↔tool activation gap is not just correlated with refusal; manipulating it changes behavior. This supports Story B: refusal is a competition between format/decision axes, not a deleted safety module.

Raw data: `linear-probes/steer_*.json`

---

## Finding 4 — Agentic SFT is not uniquely erosive

When we restore refusal by steering each model with its *own* `d_format` (removing base-alignment confound):

| Model | Refusal restored at c=1× |
|---|---|
| Base | 42% |
| Agentic FFT | 70% |
| Safety mix FFT | ~100% (saturated) |
| Chat control | **7%** |

**Takeaway:** Generic chat SFT erodes refusal *more* than agentic tool SFT. Adding ~10% safety examples to the mix makes refusal easiest to restore. The problem is not "tools are special" — it's "fine-tuning broadly weakens the refusal pathway, and safety data helps."

---

## Finding 5 — Delta crosscoder: helpfulness drowns safety (WebRL)

We compared base vs WebRL-SFT (web navigation agent data) using a delta crosscoder at layer 16.

**Behavioral effect:** AgentHarm harmful refusal 87.5% → 15.9% (−71.6 pp).

**Internal effect:**

- Mean Δβ = **+0.46** across 24,846 active sparse features — the whole dictionary shifted positive, not one feature.
- Top-100 changed features fire on chat templates, code Q&A, and technical help — not obvious "refusal" semantics.
- **20 sign-flips:** features the base model was *suppressing* that SFT now *activates*. One candidate (feature 18649) fires on jailbreak persona prompts (TranslatorBot-style). Not yet tested causally.

**Takeaway:** WebRL did not delete a safety circuit. It turned up generic "helpful assistant" features everywhere. On harmful prompts, amplified helpfulness out-competes refusal. This disagrees with the single "toxic persona feature" story from emergent misalignment work.

Raw data: `delta-crosscoder/SUMMARY.md`, `delta-crosscoder/summary.json`, plots in same folder.

---

## What we still don't know

- Whether crosscoder feature 18649 *causes* the behavioral drop (needs steering/ablation).
- Chat-format refusal rates for the three EHR fine-tunes (only base is measured in chat vs tool).
- Whether a narrow safety circuit exists but is invisible in top-|Δβ| ranking because validation data had no harmful prompts.

---

## File guide

| Directory | Contents |
|---|---|
| `behavioral/` | WebRL AgentHarm eval JSONs + layer sanity-check stats |
| `linear-probes/` | Probe summaries, steering results, refusal labels |
| `delta-crosscoder/` | Crosscoder rankings, interpretation docs, histogram/scatter plots |
