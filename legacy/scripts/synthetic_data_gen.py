from __future__ import annotations

import json
import random
import time
from typing import Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

client = OpenAI()

GEN_TEMPLATE = """
You are generating synthetic preference pairs in the style of Anthropic HH.

Create ONE example with:
1) "prompt": a realistic user message (1–3 sentences). The prompt should be everyday and conversational.
2) "chosen": the better assistant response.
3) "rejected": a worse assistant response to the same prompt.

Style requirements:
- The assistant should be generally helpful, polite, and clear.
- Responses should be natural language paragraphs (optionally short bullet points).
- Avoid meta-commentary (no talk about being an AI model, training, datasets, benchmarks, or evaluation).
- Avoid overly fancy formatting.

Quality gap:
- The "chosen" response should be clearly better, but the "rejected" response must still be plausible.
- Make the "rejected" response worse in 1–2 of these ways:
  * less helpful (too short, missing key steps)
  * less clear (disorganized, vague)
  * mild factual error or shaky assumption
  * slightly worse tone (a bit curt or preachy)
  * ignores an important constraint from the prompt
- Do NOT make the rejected response obviously terrible, nonsensical, or unsafe.

Content constraints:
- Keep content safe: no explicit sexual content, no violence instructions, no illegal wrongdoing assistance.
- Avoid extremist content, self-harm, or weapon instructions.
- It's okay to include benign refusals if the prompt requests something disallowed, but keep it rare.

Important for speed:
- Keep responses reasonably concise. Avoid very long answers.

Return ONLY valid JSON with exactly these keys:
{
  "prompt": "...",
  "chosen": "...",
  "rejected": "..."
}
""".strip()


def _call_once(model: str, temperature: float, max_tokens: int) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": GEN_TEMPLATE}],
        temperature=temperature,
        max_tokens=max_tokens,                 # caps output length (responses), not prompt quality
        response_format={"type": "json_object"}  # forces valid JSON (kills retries)
    )
    return resp.choices[0].message.content


def generate_single_pair(
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 650,   # adjust: higher = longer responses, slower; lower = faster
    max_retries: int = 0,    # with json_object, retries are usually unnecessary
) -> Dict[str, str]:
    last_err = None
    for _ in range(max_retries + 1):
        try:
            text = _call_once(model=model, temperature=temperature, max_tokens=max_tokens)
            obj = json.loads(text)
            if not all(k in obj and isinstance(obj[k], str) for k in ("prompt", "chosen", "rejected")):
                raise ValueError(f"Invalid output keys/types: {obj.keys()}")
            return {
                "prompt": obj["prompt"].strip(),
                "chosen": obj["chosen"].strip(),
                "rejected": obj["rejected"].strip(),
            }
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to generate pair: {last_err}")


def generate_to_jsonl_concurrent(
    n: int,
    model: str,
    out_path: str,
    seed: int = 0,
    temperature: float = 0.7,
    max_tokens: int = 650,
    workers: int = 8,
    log_every: int = 25,
):
    random.seed(seed)
    t0 = time.time()
    done = 0

    print(f"Starting generation: n={n} workers={workers} model={model} max_tokens={max_tokens}")

    with open(out_path, "w") as f, ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(generate_single_pair, model, temperature, max_tokens, 0)
            for _ in range(n)
        ]

        for fut in as_completed(futures):
            item = fut.result()
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            f.flush()

            done += 1
            if done == 1 or done % log_every == 0:
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-9)
                eta = (n - done) / max(rate, 1e-9)
                print(f"[{done:>4}/{n}] rate={rate:.2f}/s  ETA={eta/60:.1f} min")

    print(f"\nDone. Wrote {n} examples to {out_path}")


if __name__ == "__main__":
    NUM_ITEMS = 400
    MODEL = "gpt-4o-mini"
    OUT_PATH = "data/synthetic_hh_like.jsonl"

    # If you get rate-limited, drop workers to 4.
    generate_to_jsonl_concurrent(
        n=NUM_ITEMS,
        model=MODEL,
        out_path=OUT_PATH,
        seed=0,
        temperature=0.7,
        max_tokens=650,
        workers=8,
        log_every=25,
    )
