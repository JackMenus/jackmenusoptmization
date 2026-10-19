"""Candidate providers: where new operator/algorithm candidates come from.

All providers return :class:`Proposal` objects.  Offline providers:

* :class:`GrammarProvider`       - fresh DSL operators / algorithm compositions from the typed grammar.
* :class:`EvolutionProvider`     - mutation and crossover of retained candidates (records parents + edits).
* :class:`CodeTemplateProvider`  - executable Python operators rendered from the code-fragment grammar
                                   (only used when OS isolation is available).

Optional:

* :class:`LLMProvider` - a language-model adapter behind the :class:`LLMClient`
  protocol.  It is off by default, every call is charged to the development
  budget's ``model_calls`` category, and its outputs are treated exactly like
  any other untrusted proposal (static checks, validation, benchmarking).
  :class:`AnthropicClient` is one concrete client; :class:`ScriptedLLMClient`
  replays canned responses for tests and offline demos.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..dsl.spec import DEFAULT_OPERATOR, check_algorithm, check_operator
from .code_operators import CodeTemplateGenerator, render_mutation
from .grammar import (Bias, OperatorGrammar, crossover_algorithms, mutate_algorithm, mutate_operator, valid_algorithm,
                      valid_operator)


@dataclass
class Proposal:
    kind: str  # "operator" | "algorithm"
    representation: str  # "dsl" | "python"
    source: object  # dict (DSL) or str (python source)
    provider: str
    rationale: str = ""
    operation: str = "created"
    parents: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)  # e.g. code-grammar point, declared types


@dataclass
class ProposalContext:
    types: set
    bias: Bias
    retained: list[dict]  # registry records of currently retained candidates (best first)
    features: dict
    generation: int
    ledger: object = None


class Provider(Protocol):
    name: str

    def propose(self, n: int, ctx: ProposalContext, rng: np.random.Generator) -> list[Proposal]: ...


class GrammarProvider:
    name = "grammar"

    def __init__(self, operator_share: float = 0.4, max_tries: int = 20) -> None:
        self.operator_share = operator_share
        self.max_tries = max_tries

    def propose(self, n, ctx, rng):
        g = OperatorGrammar(ctx.types, ctx.bias)
        out = []
        for k in range(n):
            if rng.random() < self.operator_share:
                for _ in range(self.max_tries):
                    e = g.operator(rng)
                    if valid_operator(e, ctx.types):
                        out.append(Proposal("operator", "dsl", e, self.name, "random draw from the typed operator grammar"
                                            + (f"; bias: {'; '.join(ctx.bias.notes)}" if ctx.bias.notes else "")))
                        break
            else:
                for _ in range(self.max_tries):
                    a = g.algorithm(rng, name=f"grammar-g{ctx.generation}-{k}")
                    if valid_algorithm(a, ctx.types):
                        out.append(Proposal("algorithm", "dsl", a, self.name, "random composition from the algorithm grammar"
                                            + (f"; bias: {'; '.join(ctx.bias.notes)}" if ctx.bias.notes else "")))
                        break
        return out


class EvolutionProvider:
    name = "evolution"

    def __init__(self, p_crossover: float = 0.25, max_tries: int = 20) -> None:
        self.p_crossover = p_crossover
        self.max_tries = max_tries

    def propose(self, n, ctx, rng):
        dsl = [r for r in ctx.retained if r["representation"] == "dsl"]
        if not dsl:
            return []
        g = OperatorGrammar(ctx.types, ctx.bias)
        out = []
        for k in range(n):
            # rank-biased parent choice (retained list is sorted best first)
            w = 1.0 / (1 + np.arange(len(dsl)))
            pa = dsl[int(rng.choice(len(dsl), p=w / w.sum()))]
            for _ in range(self.max_tries):
                if pa["kind"] == "algorithm":
                    algs = [r for r in dsl if r["kind"] == "algorithm" and r["id"] != pa["id"]]
                    if algs and rng.random() < self.p_crossover:
                        pb = algs[int(rng.integers(len(algs)))]
                        child, desc = crossover_algorithms(pa["source"], pb["source"], rng)
                        parents = [pa["id"], pb["id"]]
                    else:
                        child, desc = mutate_algorithm(pa["source"], rng, g)
                        parents = [pa["id"]]
                    child["name"] = f"evo-g{ctx.generation}-{k}"
                    if valid_algorithm(child, ctx.types):
                        out.append(Proposal("algorithm", "dsl", child, self.name, f"edit of retained candidate: {desc}",
                                            desc, parents))
                        break
                else:
                    child, desc = mutate_operator(pa["source"], rng, g)
                    if valid_operator(child, ctx.types):
                        out.append(Proposal("operator", "dsl", child, self.name, f"edit of retained operator: {desc}",
                                            desc, [pa["id"]]))
                        break
        return out


class CodeTemplateProvider:
    name = "code_template"

    def __init__(self) -> None:
        self.gen = CodeTemplateGenerator()

    def propose(self, n, ctx, rng):
        out = []
        code_parents = [r for r in ctx.retained if r["representation"] == "python" and r.get("meta", {}).get("grammar")]
        for _ in range(n):
            if code_parents and rng.random() < 0.6:
                pa = code_parents[int(rng.integers(len(code_parents)))]
                gpt = self.gen.mutate(pa["meta"]["grammar"], rng)
                parents, op = [pa["id"]], "mutated code-grammar point"
            else:
                gpt = self.gen.sample(rng)
                parents, op = [], "sampled code-grammar point"
            out.append(Proposal("operator", "python", render_mutation(gpt), self.name,
                                f"Python mutation operator rendered from code fragments {gpt}", op, parents,
                                {"grammar": gpt, "types": self.gen.types_changed(gpt)}))
        return out


# ---------------------------------------------------------------------------
# optional language-model integration
# ---------------------------------------------------------------------------

class LLMClient(Protocol):
    def complete(self, prompt: str) -> str: ...


class ScriptedLLMClient:
    """Deterministic stand-in that replays canned responses (tests, offline demonstrations)."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def complete(self, prompt: str) -> str:
        r = self.responses[self.calls % len(self.responses)]
        self.calls += 1
        return r


class AnthropicClient:
    """Claude Messages API client (requires the ``anthropic`` package and credentials).

    Uses ``claude-opus-5-5`` by default with server-side refusal fallbacks
    enabled; the text blocks of the response are returned verbatim.
    """

    def __init__(self, model: str = "claude-opus-5-5", max_tokens: int = 16000, effort: str = "medium") -> None:
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("the optional 'anthropic' package is not installed (pip install anthropic)") from exc
        import anthropic
        self.client = anthropic.Anthropic()
        self.model, self.max_tokens, self.effort = model, max_tokens, effort

    def complete(self, prompt: str) -> str:
        resp = self.client.beta.messages.create(
            model=self.model, max_tokens=self.max_tokens, output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            messages=[{"role": "user", "content": prompt}],
        )
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"model declined the request: {getattr(resp, 'stop_details', None)}")
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


PROMPT = """You design search operators for a black-box optimisation platform.
Write ONE JSON object and nothing else, for a new candidate in this restricted language.

Algorithm form: {{"dsl_version": 1, "kind": "algorithm", "name": str, "population": int, "offspring": int,
 "init": {{"op": "lhs"|"uniform"|"opposition"}}, "repair": {{"op": "project"|"reflect"|"resample"}},
 "stages": [{{"name": str, "until": <predicate, all but last stage>, "parent_selection": {{"op": "tournament", "k": int}},
   "variation": <expr>, "survival": {{"op": "plus"|"comma"|"pairwise"|"metropolis"|"threshold", ...}},
   "step": {{"op": "constant"|"one_fifth"|"decay", ...}} }}],
 "restart": null | {{"op": "stagnation", "patience": int, "keep_best": int, "reinit": "lhs"|"around_best", "scale": float}},
 "refine": null | {{"op": "pattern", "evals_fraction": float, "step0": float}} }}
Expressions: producers {producers}; transformers {transformers};
combinators seq(children), mix(children, weights), if(cond, then, else), per_type(real=..., integer=..., ...),
prob(p, child), repeat(n, child). Predicates: stagnation/evals_fraction/success_rate/feasible_fraction/diversity
with "ge" or "lt", random(p), and/or/not.

Target problems: variable types {types}; features {features}.
Current best retained candidates (with training scores, lower is better): {retained}
Propose something structurally different from the retained candidates, and say nothing outside the JSON."""


class LLMProvider:
    name = "llm"

    def __init__(self, client: LLMClient, allow_code: bool = False) -> None:
        self.client = client
        self.allow_code = allow_code
        self.failures: list[str] = []

    def _prompt(self, ctx: ProposalContext) -> str:
        from ..dsl.spec import OPS
        prods = sorted(k for k, v in OPS.items() if v[0] == "producer")
        trans = sorted(k for k, v in OPS.items() if v[0] == "transformer")
        ret = [{"id": r["id"], "score": r.get("train_score"), "source": r["source"]} for r in ctx.retained[:3]
               if r["representation"] == "dsl"]
        return PROMPT.format(producers=prods, transformers=trans, types=sorted(ctx.types),
                             features={k: v for k, v in ctx.features.items() if not isinstance(v, (list, dict))},
                             retained=json.dumps(ret)[:6000])

    def propose(self, n, ctx, rng):
        out = []
        for _ in range(n):
            if ctx.ledger is not None:
                ctx.ledger.charge("model_calls", 1)
            try:
                text = self.client.complete(self._prompt(ctx))
                m = re.search(r"\{.*\}", text, re.S)
                obj = json.loads(m.group(0)) if m else None
            except Exception as exc:
                self.failures.append(repr(exc))
                continue
            if not isinstance(obj, dict):
                self.failures.append("no JSON object in model output")
                continue
            if obj.get("kind") == "algorithm":
                rep = check_algorithm(obj, ctx.types)
                kind = "algorithm"
            else:
                rep = check_operator(obj, ctx.types)
                kind = "operator"
            # statically invalid proposals are still recorded (and rejected) so the registry shows them
            out.append(Proposal(kind, "dsl", obj, self.name, "language-model proposal" + ("" if rep.ok else
                                f" (fails static checks: {rep.errors[:2]})")))
        return out


def default_providers(config: dict, isolation_available: bool) -> tuple[list, list[str]]:
    """Build the provider list from a development config; returns (providers, notes)."""
    notes = []
    provs: list = [GrammarProvider(config.get("operator_share", 0.4)), EvolutionProvider()]
    if config.get("code_generation", True):
        if isolation_available:
            provs.append(CodeTemplateProvider())
        else:
            notes.append("code generation disabled: operating-system isolation unavailable; using the declarative DSL only")
    llm = config.get("llm")
    if llm:
        if llm.get("provider") == "anthropic":
            try:
                provs.append(LLMProvider(AnthropicClient(llm.get("model", "claude-opus-5-5"))))
            except Exception as exc:
                notes.append(f"LLM provider not enabled: {exc}")
        elif llm.get("provider") == "scripted":
            provs.append(LLMProvider(ScriptedLLMClient(llm["responses"])))
    return provs, notes


__all__ = ["Proposal", "ProposalContext", "GrammarProvider", "EvolutionProvider", "CodeTemplateProvider",
           "LLMProvider", "LLMClient", "AnthropicClient", "ScriptedLLMClient", "default_providers", "DEFAULT_OPERATOR"]
