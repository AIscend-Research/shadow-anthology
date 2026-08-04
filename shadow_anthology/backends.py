"""Backends that expose the sampler's distribution, not just its output.

The whole method rests on one requirement: at every position we must see the
tokens that *lost*, not only the token that won. That rules out most chat APIs.

  * `HFBackend`      -- local `transformers` model. Exact, full top-k, seedable,
                        supports forced-prefix continuation, so it is the
                        reference backend for anything in the paper.
  * `OpenAICompatBackend` -- any endpoint serving `logprobs` + `top_logprobs`
                        (OpenAI, vLLM, llama.cpp server, TGI, ...). Capped at
                        the server's top_logprobs limit (commonly 20).
  * `MockBackend`    -- deterministic toy sampler, zero dependencies. Used by
                        the tests and by `shadow demo` so the pipeline is
                        runnable and inspectable without a GPU.

NOT SUPPORTED, and worth stating plainly: the Anthropic Messages API does not
return logprobs or runner-up tokens, so Claude traces cannot be captured this
way. See `AnthropicUnsupported` below and the Limitations section of the paper.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .trace import Candidate, GenerationTrace, TokenStep

DEFAULT_TOP_K = 20


@runtime_checkable
class Backend(Protocol):
    """Minimal contract every backend must satisfy."""

    name: str
    model: str
    supports_forced_prefix: bool
    """True if the backend can be made to resume generation from an arbitrary
    token sequence. Required for the branching anthology (`anthology.py`);
    shadow reconstruction works without it."""

    def generate_trace(
        self,
        prompt: str,
        *,
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        candidates: int = DEFAULT_TOP_K,
        seed: int | None = None,
        system: str | None = None,
        stop: Sequence[str] | None = None,
    ) -> GenerationTrace: ...

    def continue_from(
        self,
        trace: GenerationTrace,
        upto: int,
        forced: Candidate,
        *,
        max_tokens: int = 128,
        **kwargs: Any,
    ) -> GenerationTrace:
        """Re-run generation with `trace`'s first `upto` tokens fixed, then
        `forced`, then free continuation. Raises if not supported."""
        ...


class BackendUnsupported(RuntimeError):
    pass


class APIRequestFailed(RuntimeError):
    """An HTTP error that carries the server's own explanation.

    Written after a debugging session in which a bare `404 Not Found` sent us
    hunting the endpoint URL, when the body said `model not found` and the
    real problem was an invalid key two calls away. The status code alone is
    close to useless on these APIs; the body is where the answer is.

    Note in particular that Fireworks resolves the model *before* checking
    auth, so an unknown model returns 404 whether or not your key is valid --
    which is why the hint below never claims the key is fine.
    """

    HINTS = {
        401: (
            "The key was rejected. Check FIREWORKS_API_KEY is set to the FULL "
            "key (they are long; a truncated copy is the usual cause), and that "
            "it has not been revoked."
        ),
        403: "The key is valid but not permitted to use this model or account.",
        404: (
            "The model id was not found, is not deployed, or your account "
            "cannot reach it. NOTE: this check runs BEFORE authentication, so a "
            "404 here does not tell you whether your key is valid -- verify "
            "that separately against a models/list endpoint."
        ),
        429: "Rate limited. Lower --concurrency or retry.",
    }

    def __init__(self, status: int, body: str, model: str = "?") -> None:
        self.status = status
        self.body = body
        detail = body.strip()
        try:
            import json as _json

            parsed = _json.loads(body)
            detail = parsed.get("error", {}).get("message", detail) or detail
        except Exception:
            pass
        hint = self.HINTS.get(status, "")
        super().__init__(
            f"HTTP {status} from the API (model={model!r}): {detail[:400]}"
            + (f"\n  -> {hint}" if hint else "")
        )


class AnthropicUnsupported:
    """Placeholder that documents why Claude cannot be traced this way.

    Kept as a real object so the failure is explicit and greppable rather than
    a mysterious absence. If the Messages API ever exposes top-k logprobs, this
    is the seam to implement against.
    """

    name = "anthropic"
    supports_forced_prefix = False

    def __init__(self, *_: Any, **__: Any) -> None:
        raise BackendUnsupported(
            "The Anthropic Messages API does not expose logprobs or runner-up "
            "tokens, so a sampler trace cannot be reconstructed from it. Use "
            "the 'hf' backend (local weights) or 'openai' backend (an endpoint "
            "serving top_logprobs). See README 'Which models can be traced'."
        )


# --------------------------------------------------------------------------
# Local transformers
# --------------------------------------------------------------------------


class HFBackend:
    """Local causal LM with a hand-rolled sampling loop.

    We do not use `model.generate()`: we need the distribution *the sampler
    actually drew from* at each step, after temperature and truncation, which
    the convenience API discards. The loop below is the point of the project.
    """

    name = "hf"
    supports_forced_prefix = True

    def __init__(
        self,
        model: str = "gpt2",
        device: str | None = None,
        dtype: str | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise BackendUnsupported(
                "HFBackend needs torch + transformers: pip install 'shadow-anthology[hf]'"
            ) from exc

        self._torch = torch
        self.model = model
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
        self.device = device
        kwargs: dict[str, Any] = {}
        if dtype:
            kwargs["torch_dtype"] = getattr(torch, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.lm = AutoModelForCausalLM.from_pretrained(model, **kwargs).to(device)
        self.lm.eval()

    # -- helpers ------------------------------------------------------------

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _apply_chat_template(self, prompt: str, system: str | None) -> str:
        tpl = getattr(self.tokenizer, "chat_template", None)
        if not tpl:
            return prompt if system is None else f"{system}\n\n{prompt}"
        msgs = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    def _step_distribution(
        self, logits, temperature: float, top_p: float, top_k: int, n_cand: int
    ):
        """Return (candidate_ids, sampling_logprobs, raw_logprobs) for one step.

        Order of operations mirrors standard sampling: temperature, then top-k,
        then nucleus. Candidates are read off *after* truncation, so a token the
        sampler could never have produced is never reported as a runner-up ---
        that would be a lie about what almost happened.
        """
        torch = self._torch
        raw_lp = torch.log_softmax(logits.float(), dim=-1)

        scaled = logits.float() / max(temperature, 1e-6)
        if top_k and top_k > 0:
            kth = torch.topk(scaled, min(top_k, scaled.shape[-1])).values[..., -1:]
            scaled = scaled.masked_fill(scaled < kth, float("-inf"))
        if top_p and top_p < 1.0:
            srt, idx = torch.sort(scaled, descending=True)
            cum = torch.cumsum(torch.softmax(srt, dim=-1), dim=-1)
            # keep everything strictly before the crossing, plus the crosser
            drop = cum - torch.softmax(srt, dim=-1) > top_p
            srt = srt.masked_fill(drop, float("-inf"))
            scaled = torch.full_like(scaled, float("-inf")).scatter(-1, idx, srt)

        samp_lp = torch.log_softmax(scaled, dim=-1)
        finite = int(torch.isfinite(samp_lp).sum().item())
        k = max(1, min(n_cand, finite))
        top = torch.topk(samp_lp, k)
        return top.indices.tolist(), top.values.tolist(), raw_lp[top.indices].tolist()

    def _run(
        self,
        input_ids: list[int],
        *,
        forced: list[int],
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        candidates: int,
        rng: "random.Random",
        stop: Sequence[str] | None,
    ) -> tuple[list[TokenStep], str]:
        torch = self._torch
        steps: list[TokenStep] = []
        ids = list(input_ids)
        past = None
        cur = torch.tensor([ids], device=self.device)
        eos = self.tokenizer.eos_token_id

        with torch.no_grad():
            for i in range(max_tokens):
                out = self.lm(cur, past_key_values=past, use_cache=True)
                past = out.past_key_values
                logits = out.logits[0, -1, :]

                cand_ids, samp_lps, raw_lps = self._step_distribution(
                    logits, temperature, top_p, top_k, candidates
                )
                cands = [
                    Candidate(
                        token_id=t,
                        text=self.tokenizer.decode([t]),
                        logprob=lp,
                        raw_logprob=rlp,
                    )
                    for t, lp, rlp in zip(cand_ids, samp_lps, raw_lps)
                ]

                if i < len(forced):
                    nxt = forced[i]
                else:
                    probs = [math.exp(lp) for lp in samp_lps]
                    z = sum(probs)
                    nxt = _weighted_choice(cand_ids, [p / z for p in probs], rng)

                if not any(c.token_id == nxt for c in cands):
                    # A forced token outside the truncated support: keep it, and
                    # record it honestly at the tail rather than dropping it.
                    cands.append(
                        Candidate(
                            token_id=nxt,
                            text=self.tokenizer.decode([nxt]),
                            logprob=float(
                                torch.log_softmax(logits.float(), dim=-1)[nxt].item()
                            ),
                            raw_logprob=float(
                                torch.log_softmax(logits.float(), dim=-1)[nxt].item()
                            ),
                        )
                    )
                    cands.sort(key=lambda c: -c.logprob)

                rank = next(j for j, c in enumerate(cands) if c.token_id == nxt)
                steps.append(
                    TokenStep(
                        index=len(steps),
                        chosen=cands[rank],
                        candidates=cands,
                        chosen_rank=rank,
                    )
                )
                ids.append(nxt)
                cur = torch.tensor([[nxt]], device=self.device)

                if eos is not None and nxt == eos:
                    break
                text_so_far = "".join(s.chosen.text for s in steps)
                if stop and any(s in text_so_far for s in stop):
                    break

        text = "".join(s.chosen.text for s in steps)
        if stop:
            for s in stop:
                if s in text:
                    text = text.split(s)[0]
        return steps, text

    # -- API ----------------------------------------------------------------

    def generate_trace(
        self,
        prompt: str,
        *,
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        candidates: int = DEFAULT_TOP_K,
        seed: int | None = None,
        system: str | None = None,
        stop: Sequence[str] | None = None,
    ) -> GenerationTrace:
        rendered = self._apply_chat_template(prompt, system)
        ids = self._encode(rendered)
        rng = random.Random(seed)
        if seed is not None:
            self._torch.manual_seed(seed)
        steps, text = self._run(
            ids,
            forced=[],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            candidates=candidates,
            rng=rng,
            stop=stop,
        )
        return GenerationTrace(
            text=text,
            steps=steps,
            prompt=prompt,
            system=system,
            model=self.model,
            backend=self.name,
            seed=seed,
            params={
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "candidates": candidates,
                "max_tokens": max_tokens,
            },
        )

    def continue_from(
        self,
        trace: GenerationTrace,
        upto: int,
        forced: Candidate,
        *,
        max_tokens: int = 128,
        seed: int | None = None,
        **kwargs: Any,
    ) -> GenerationTrace:
        p = trace.params
        rendered = self._apply_chat_template(trace.prompt, trace.system)
        prompt_ids = self._encode(rendered)
        prefix = [s.chosen.token_id for s in trace.steps[:upto]] + [forced.token_id]
        rng = random.Random(seed if seed is not None else trace.seed)
        if seed is not None:
            self._torch.manual_seed(seed)
        steps, text = self._run(
            prompt_ids,
            forced=prefix,
            max_tokens=max(max_tokens, len(prefix)),
            temperature=kwargs.get("temperature", p.get("temperature", 1.0)),
            top_p=kwargs.get("top_p", p.get("top_p", 1.0)),
            top_k=kwargs.get("top_k", p.get("top_k", 0)),
            candidates=kwargs.get("candidates", p.get("candidates", DEFAULT_TOP_K)),
            rng=rng,
            stop=kwargs.get("stop"),
        )
        return GenerationTrace(
            text=text,
            steps=steps,
            prompt=trace.prompt,
            system=trace.system,
            model=self.model,
            backend=self.name,
            seed=seed,
            params=dict(p),
            meta={"branch_at": upto, "branch_token": forced.text},
        )


# --------------------------------------------------------------------------
# OpenAI-compatible endpoints
# --------------------------------------------------------------------------


class OpenAICompatBackend:
    """Any server exposing `logprobs` + `top_logprobs` on completions.

    Two honest caveats, both consequences of reading a trace back rather than
    driving the sampler ourselves:

      1. `top_logprobs` is capped server-side (20 on OpenAI), so deep-rank
         anthologies are bounded by that cap.
      2. Returned logprobs are the *model's* distribution, not the tempered
         one the sampler drew from, so ranks reflect model preference. At
         temperature 1.0 these coincide; below it they can disagree, and
         `Candidate.logprob` should then be read as raw model preference.
    """

    name = "openai"
    supports_forced_prefix = True  # via the /completions prefix trick

    def __init__(
        self,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        endpoint: str = "chat",
    ) -> None:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise BackendUnsupported(
                "OpenAICompatBackend needs httpx: pip install 'shadow-anthology[openai]'"
            ) from exc
        import os

        self._httpx = httpx
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.endpoint = endpoint
        if endpoint != "completions":
            self.supports_forced_prefix = False

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        r = self._httpx.post(
            f"{self.base_url}{path}", json=body, headers=headers, timeout=120.0
        )
        if r.status_code >= 400:
            raise APIRequestFailed(r.status_code, r.text, body.get("model", "?"))
        return r.json()

    def _steps_from_logprobs(self, content: list[dict[str, Any]]) -> list[TokenStep]:
        """Build steps from an OpenAI-shaped `logprobs.content` array.

        Where the server reports `sampling_logprob` (Fireworks does), we prefer
        it for ranking: that is the tempered distribution the sampler actually
        drew from, which is the quantity this project is about. `logprob` is
        kept as `raw_logprob`. Servers that report only `logprob` fall back to
        it for both, and `meta.logprobs_are_raw` records which happened.
        """
        steps: list[TokenStep] = []
        for i, item in enumerate(content):
            tops = list(item.get("top_logprobs") or [])

            # Scale consistency. The chosen entry often carries
            # `sampling_logprob` while the top_logprobs entries carry only the
            # raw `logprob`. Ranking a tempered value against raw ones compares
            # different units and corrupts chosen_rank, so use the tempered
            # scale only when EVERY entry at this position has it.
            use_sampling = all(
                x.get("sampling_logprob") is not None for x in [item] + tops
            )
            cands = [_candidate_from(t, use_sampling) for t in tops]
            chosen = _candidate_from(item, use_sampling)

            # Dedup by token id where the server gives one, falling back to
            # surface text. Comparing logprobs here (the previous approach)
            # appended a second copy of the chosen token whenever the two
            # sources reported it on different scales -- which then surfaced as
            # its own "runner-up", identical on the page, at 76% of positions.
            def same(a: Candidate, b: Candidate) -> bool:
                if a.token_id >= 0 and b.token_id >= 0:
                    return a.token_id == b.token_id
                return a.text == b.text

            if not any(same(c, chosen) for c in cands):
                cands.append(chosen)

            cands.sort(key=lambda c: -c.logprob)
            rank = next(
                (j for j, c in enumerate(cands) if same(c, chosen)),
                0,
            )
            steps.append(
                TokenStep(index=i, chosen=cands[rank], candidates=cands, chosen_rank=rank)
            )
        return steps

    max_top_logprobs = 20
    """Server-side cap on retained alternatives. Requests above it are clamped
    and the effective value is recorded in the trace params, so a trace never
    silently claims more candidate depth than the server returned."""

    def generate_trace(
        self,
        prompt: str,
        *,
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        candidates: int = DEFAULT_TOP_K,
        seed: int | None = None,
        system: str | None = None,
        stop: Sequence[str] | None = None,
    ) -> GenerationTrace:
        eff = min(candidates, self.max_top_logprobs)
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "logprobs": True,
            "top_logprobs": eff,
        }
        if seed is not None:
            body["seed"] = seed
        if stop:
            body["stop"] = list(stop)

        if self.endpoint == "completions":
            body["prompt"] = prompt if system is None else f"{system}\n\n{prompt}"
            # `logprobs` stays boolean here. Sending the integer form alongside
            # `top_logprobs` is rejected ("logprobs must be a True if
            # top_logprobs is set"), and the boolean form is also the one that
            # returns `sampling_logprob`, which is the field we actually want.
            data = self._post("/completions", body)
            choice = data["choices"][0]
            lp = choice["logprobs"]
            if isinstance(lp, dict) and "content" in lp:
                # OpenAI-shaped response (what the boolean form returns).
                content = lp["content"]
            else:
                # Legacy shape: parallel arrays, `top_logprobs` a per-position
                # token->logprob mapping. Kept for servers that only do this.
                tops = lp.get("top_logprobs") or [None] * len(lp["tokens"])
                content = [
                    {
                        "token": t,
                        "logprob": l,
                        "top_logprobs": [
                            {"token": k, "logprob": v} for k, v in (d or {}).items()
                        ],
                    }
                    for t, l, d in zip(lp["tokens"], lp["token_logprobs"], tops)
                ]
            text = choice["text"]
        else:
            msgs = ([{"role": "system", "content": system}] if system else []) + [
                {"role": "user", "content": prompt}
            ]
            body["messages"] = msgs
            data = self._post("/chat/completions", body)
            choice = data["choices"][0]
            content = choice["logprobs"]["content"]
            text = choice["message"]["content"]

        return GenerationTrace(
            text=text,
            steps=self._steps_from_logprobs(content),
            prompt=prompt,
            system=system,
            model=self.model,
            backend=self.name,
            seed=seed,
            params={
                "temperature": temperature,
                "top_p": top_p,
                "candidates": eff,
                "candidates_requested": candidates,
                "max_tokens": max_tokens,
                "endpoint": self.endpoint,
            },
            meta={
                "logprobs_are_raw": not any(
                    "sampling_logprob" in i for i in content
                ),
                "server_capped_candidates": eff < candidates,
            },
        )

    def continue_from(
        self,
        trace: GenerationTrace,
        upto: int,
        forced: Candidate,
        *,
        max_tokens: int = 128,
        **kwargs: Any,
    ) -> GenerationTrace:
        if not self.supports_forced_prefix:
            raise BackendUnsupported(
                "Forced-prefix continuation needs endpoint='completions'; chat "
                "endpoints cannot resume mid-message."
            )
        prefix = trace.prefix_text(upto) + forced.text
        base = trace.prompt if trace.system is None else f"{trace.system}\n\n{trace.prompt}"
        sub = self.generate_trace(
            base + prefix,
            max_tokens=max_tokens,
            temperature=kwargs.get("temperature", trace.params.get("temperature", 1.0)),
            top_p=kwargs.get("top_p", trace.params.get("top_p", 1.0)),
            candidates=kwargs.get("candidates", trace.params.get("candidates", 20)),
        )
        sub.text = prefix + sub.text
        sub.prompt = trace.prompt
        sub.system = trace.system
        sub.meta = {"branch_at": upto, "branch_token": forced.text, "prefix_untraced": True}
        return sub


class FireworksBackend(OpenAICompatBackend):
    """Fireworks AI --- the best hosted option for this method.

    Two things it gives you that the OpenAI API does not:

      * `sampling_logprob` alongside `logprob`, i.e. the *tempered* distribution
        the sampler actually drew from. That is precisely the quantity this
        project reconstructs, so ranks here reflect the sampler's real
        preferences rather than the model's untempered ones.
      * a `/completions` endpoint over open-weights models, so a forced prefix
        is just a string --- branching anthologies work.

    The binding constraint is depth: `top_logprobs` is capped at the
    deployment's `--max-logprobs`, which defaults to **5**. That is the chosen
    token plus ~4 alternatives, enough for rank-1..3 shadows and not enough for
    deep-rank work. Raise it on a dedicated deployment and pass
    `max_top_logprobs` to match; requests above the cap are clamped, never
    silently honoured.
    """

    name = "fireworks"
    max_top_logprobs = 5

    def __init__(
        self,
        model: str = "accounts/fireworks/models/llama-v3p1-8b-instruct",
        base_url: str = "https://api.fireworks.ai/inference/v1",
        api_key: str | None = None,
        endpoint: str = "completions",
        max_top_logprobs: int | None = None,
    ) -> None:
        import os

        super().__init__(
            model=model,
            base_url=base_url,
            api_key=api_key or os.environ.get("FIREWORKS_API_KEY", ""),
            endpoint=endpoint,
        )
        if max_top_logprobs is not None:
            self.max_top_logprobs = max_top_logprobs
        if not self.api_key:
            raise BackendUnsupported(
                "No Fireworks API key. Set FIREWORKS_API_KEY, or pass api_key=."
            )


# --------------------------------------------------------------------------
# Dependency-free toy sampler
# --------------------------------------------------------------------------

_MOCK_VOCAB = [
    " light", " water", " stone", " bone", " glass", " ash", " root", " salt",
    " river", " window", " morning", " winter", " hunger", " engine", " thread",
    " shadow", " mouth", " field", " hour", " rain", " iron", " bird", " smoke",
    " door", " grief", " honey", " needle", " lantern", " wire", " snow",
    " and", " the", " of", " a", " in", " like", " through", " against", " that",
    " is", " was", " turns", " breaks", " holds", " opens", " burns", " falls",
    " remembers", " forgets", " keeps", " leaves", " carries", " answers",
    ",", ".", "\n", "\n\n", " —",
]


class MockBackend:
    """A deterministic pseudo-model. Not a language model; a fixture.

    Its distributions are generated by hashing the local context, which gives
    stable, seed-reproducible traces with realistic shape (a peaked head, a
    long tail, occasional near-ties) so every downstream component can be
    tested and demonstrated without weights.
    """

    name = "mock"
    supports_forced_prefix = True

    def __init__(self, model: str = "mock-poet-v1", vocab: Sequence[str] | None = None):
        self.model = model
        self.vocab = list(vocab) if vocab else list(_MOCK_VOCAB)

    def _dist(self, context: Sequence[int], salt: int) -> list[tuple[int, float]]:
        key = f"{salt}:{','.join(map(str, context[-3:]))}".encode()
        h = hashlib.blake2b(key, digest_size=32).digest()
        scores = []
        for i, _ in enumerate(self.vocab):
            b = h[i % len(h)] ^ ((i * 37) & 0xFF)
            scores.append((i, (b / 255.0) * 6.0 - 3.0))
        scores.sort(key=lambda kv: -kv[1])
        z = sum(math.exp(s) for _, s in scores)
        return [(i, s - math.log(z)) for i, s in scores]

    def _run(
        self, forced: list[int], max_tokens: int, candidates: int, seed: int | None, salt: int
    ) -> list[TokenStep]:
        rng = random.Random(seed)
        ctx: list[int] = []
        steps: list[TokenStep] = []
        for i in range(max_tokens):
            dist = self._dist(ctx, salt)[:candidates]
            cands = [
                Candidate(token_id=t, text=self.vocab[t], logprob=lp, raw_logprob=lp)
                for t, lp in dist
            ]
            if i < len(forced):
                nxt = forced[i]
                if not any(c.token_id == nxt for c in cands):
                    cands.append(Candidate(nxt, self.vocab[nxt], -12.0, -12.0))
                    cands.sort(key=lambda c: -c.logprob)
            else:
                ps = [math.exp(c.logprob) for c in cands]
                z = sum(ps)
                nxt = _weighted_choice([c.token_id for c in cands], [p / z for p in ps], rng)
            rank = next(j for j, c in enumerate(cands) if c.token_id == nxt)
            steps.append(
                TokenStep(index=i, chosen=cands[rank], candidates=cands, chosen_rank=rank)
            )
            ctx.append(nxt)
        return steps

    def generate_trace(
        self,
        prompt: str,
        *,
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        candidates: int = DEFAULT_TOP_K,
        seed: int | None = None,
        system: str | None = None,
        stop: Sequence[str] | None = None,
    ) -> GenerationTrace:
        salt = int(hashlib.blake2b(prompt.encode(), digest_size=4).hexdigest(), 16)
        steps = self._run([], max_tokens, candidates, seed, salt)
        return GenerationTrace(
            text="".join(s.chosen.text for s in steps),
            steps=steps,
            prompt=prompt,
            system=system,
            model=self.model,
            backend=self.name,
            seed=seed,
            params={
                "temperature": temperature,
                "top_p": top_p,
                "candidates": candidates,
                "max_tokens": max_tokens,
                "salt": salt,
            },
        )

    def continue_from(
        self,
        trace: GenerationTrace,
        upto: int,
        forced: Candidate,
        *,
        max_tokens: int = 128,
        seed: int | None = None,
        **kwargs: Any,
    ) -> GenerationTrace:
        prefix = [s.chosen.token_id for s in trace.steps[:upto]] + [forced.token_id]
        steps = self._run(
            prefix,
            max(max_tokens, len(prefix)),
            trace.params.get("candidates", DEFAULT_TOP_K),
            seed if seed is not None else trace.seed,
            trace.params.get("salt", 0),
        )
        return GenerationTrace(
            text="".join(s.chosen.text for s in steps),
            steps=steps,
            prompt=trace.prompt,
            system=trace.system,
            model=self.model,
            backend=self.name,
            seed=seed,
            params=dict(trace.params),
            meta={"branch_at": upto, "branch_token": forced.text},
        )


# --------------------------------------------------------------------------

_REGISTRY = {
    "hf": HFBackend,
    "openai": OpenAICompatBackend,
    "fireworks": FireworksBackend,
    "mock": MockBackend,
    "anthropic": AnthropicUnsupported,
}


def _candidate_from(item: Mapping[str, Any], use_sampling: bool = True) -> Candidate:
    """One candidate from an OpenAI/Fireworks logprobs entry.

    `sampling_logprob` (post-temperature) is what the sampler actually drew
    from and is preferred for ranking --- but only when every candidate at the
    position has it, since mixing the two scales is meaningless. The caller
    decides that and passes `use_sampling`; `logprob` is always retained as
    `raw_logprob`.
    """
    raw = float(item["logprob"])
    samp = item.get("sampling_logprob")
    tid = item.get("token_id")
    return Candidate(
        token_id=int(tid) if tid is not None else -1,
        text=item["token"],
        logprob=float(samp) if (use_sampling and samp is not None) else raw,
        raw_logprob=raw,
    )


def get_backend(kind: str, **kwargs: Any) -> Backend:
    """Construct a backend by name. Raises BackendUnsupported with a useful
    message rather than an ImportError traceback when deps are missing."""
    try:
        cls = _REGISTRY[kind]
    except KeyError:
        raise BackendUnsupported(
            f"unknown backend {kind!r}; known: {', '.join(sorted(_REGISTRY))}"
        ) from None
    return cls(**kwargs)  # type: ignore[return-value]


def _weighted_choice(items: Sequence[int], weights: Sequence[float], rng: random.Random) -> int:
    r = rng.random()
    acc = 0.0
    for it, w in zip(items, weights):
        acc += w
        if r <= acc:
            return it
    return items[-1]
