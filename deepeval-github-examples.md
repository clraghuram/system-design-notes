# DeepEval GitHub Examples: LLM Testing, Latency, Rubrics & Multi-Locale

> Research synthesized June 2026 · Sources verified via direct fetch

---

## What is DeepEval?

[DeepEval](https://github.com/confident-ai/deepeval) (by Confident AI) is a pytest-style LLM evaluation framework. Install with `pip install deepeval`. It provides 20+ built-in metrics, a `GEval` custom-rubric metric, and `EvaluationDataset` for batch testing.

---

## 1. Testing LLM Responses & Measuring Latency

### How DeepEval Tracks Latency

DeepEval does NOT have a standalone "latency metric" that scores pass/fail by response time, but it provides two hooks:

**a) `completion_time` on `LLMTestCase`** — log elapsed seconds per interaction:

```python
import time
from deepeval.test_case import LLMTestCase

start = time.time()
output = your_llm(input_prompt)
elapsed = time.time() - start

test_case = LLMTestCase(
    input=input_prompt,
    actual_output=output,
    completion_time=elapsed   # seconds — logged to Confident AI dashboard
)
```

**b) `@trace` decorators** (added 2025) — token-level streaming timestamps on spans:

```python
from deepeval.tracing import trace, TraceType

@trace(type=TraceType.LLM)
def call_llm(prompt: str) -> str:
    return openai_client.chat.completions.create(...)
```

### Best GitHub Examples — Latency & Response Testing

| Repo | File | What it does |
|------|------|--------------|
| [`vudayani/DeepEvalTests`](https://github.com/vudayani/DeepEvalTests) | `tests/` + `main.py` | Exposes a `/evaluate` POST endpoint; records G-Eval accuracy score + reasoning for Copilot LLM responses. Demonstrates API-wrapping pattern for continuous latency + quality tracking. |
| [`confident-ai/deepeval`](https://github.com/confident-ai/deepeval/blob/main/examples/tracing/test_chatbot.py) | `examples/tracing/test_chatbot.py` | Official chatbot tracing example using `@trace` decorators on LLM, embedding, retriever, and agent spans. HallucinationMetric at threshold=0.8. |
| [`confident-ai/deepeval`](https://github.com/confident-ai/deepeval/blob/main/examples/getting_started/test_example.py) | `examples/getting_started/test_example.py` | Canonical quickstart: `AnswerRelevancyMetric` + `GEval` correctness, `EvaluationDataset`, `log_hyperparameters`. |
| [`codemaker2015/deepeval-experiments`](https://github.com/codemaker2015/deepeval-experiments) | 12 experiment files | Hands-on coverage of AnswerRelevancy, G-Eval, Hallucination, Faithfulness, Multi-turn chatbot, LLM tracing, and MMLU benchmarks — each as a separate runnable test file. |
| [`meteatamel/genai-beyond-basics`](https://github.com/meteatamel/genai-beyond-basics/blob/main/samples/evaluation/deepeval/rag_eval/test_rag_triad.py) | `rag_eval/test_rag_triad.py` | Google DevRel sample: RAG Triad (AnswerRelevancy, Faithfulness, ContextualRelevancy) against a live Vertex AI + Gemini backend. Uses `assert_test()` to fail CI on regressions. |

---

## 2. Building Evaluation Rubrics with GEval

### The `Rubric` Class (canonical definition)

Source: [`confident-ai/deepeval/blob/main/deepeval/metrics/g_eval/utils.py`](https://github.com/confident-ai/deepeval/blob/main/deepeval/metrics/g_eval/utils.py)

```python
from pydantic import BaseModel, field_validator
from typing import Tuple

class Rubric(BaseModel):
    score_range: Tuple[int, int]   # e.g. (0, 3), (4, 7), (8, 10) — must be within 0–10
    expected_outcome: str          # human description of this score band
```

### GEval with Multi-Band Rubric

Source: [`patchy631/ai-engineering-hub — code_evaluation.py`](https://github.com/patchy631/ai-engineering-hub/blob/main/sonnet4-vs-o4/code_evaluation.py)

```python
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

correctness = GEval(
    name="CodeCorrectness",
    criteria="Evaluate if the code is functionally correct, handles edge cases, and implements the required functionality completely.",
    evaluation_steps=[
        "Check if the code runs without errors",
        "Verify all required functionality is implemented",
        "Test edge case handling",
    ],
    rubric=[
        Rubric(score_range=(0, 2),  expected_outcome="Code is non-functional or has critical errors"),
        Rubric(score_range=(3, 5),  expected_outcome="Code works but misses key functionality"),
        Rubric(score_range=(6, 8),  expected_outcome="Code is mostly correct with minor issues"),
        Rubric(score_range=(9, 10), expected_outcome="Code is completely correct and handles all cases"),
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
)

test_case = LLMTestCase(input=prompt, actual_output=generated_code, expected_output=reference_code)
correctness.measure(test_case)
print(correctness.score, correctness.reason)
```

### Binary Per-Criterion Rubric (alternative to holistic GEval)

Source: [`topoteretes/cognee — rubric.py`](https://github.com/topoteretes/cognee/blob/main/cognee/eval_framework/evaluation/metrics/rubric.py)

```python
# Attach rubric as metadata — each item evaluated YES/NO independently
test_case = LLMTestCase(
    input="What is the capital of France?",
    actual_output=llm_response,
    additional_metadata={
        "rubric": [
            "Response mentions Paris",
            "Response includes population figure",
            "Response does not hallucinate country boundaries",
        ]
    }
)
# score = satisfied_items / total_items (0.0–1.0)
# reason lists up to 3 failed criteria
```

### Best GitHub Examples — Rubrics

| Repo | File | Key Pattern |
|------|------|-------------|
| [`confident-ai/deepeval`](https://github.com/confident-ai/deepeval/blob/main/deepeval/metrics/g_eval/utils.py) | `g_eval/utils.py` | Canonical `Rubric` class + `validate_and_sort_rubrics()` + `calculate_weighted_summed_score()` |
| [`patchy631/ai-engineering-hub`](https://github.com/patchy631/ai-engineering-hub/blob/main/sonnet4-vs-o4/code_evaluation.py) | `sonnet4-vs-o4/code_evaluation.py` | 4-band rubric GEval across 3 metrics (correctness, readability, best practices), averaged to one score |
| [`crunchdomo/thesis-cooking-conversation-final`](https://github.com/crunchdomo/thesis-cooking-conversation-final/blob/main/evaluation/updated_deepeval_rag_metrics.py) | `evaluation/updated_deepeval_rag_metrics.py` | 4 domain-specific GEval metrics for RAG; 5-point rubric scale; overrides built-in AnswerRelevancy |
| [`LordZerror/GroundedDialogue`](https://github.com/LordZerror/GroundedDialogue/blob/main/eval/run_geval.py) | `eval/run_geval.py` | G-Eval rubric pipeline calling Groq API directly — 3 rubrics: Mode Appropriateness, No Answer Leak, Format Compliance; outputs to CSV |
| [`topoteretes/cognee`](https://github.com/topoteretes/cognee/blob/main/cognee/eval_framework/evaluation/metrics/rubric.py) | `eval_framework/evaluation/metrics/rubric.py` | Binary per-criterion rubric: YES/NO per item → fraction score; async-native; best for auditable, transparent evaluation |
| [`NirDiamant/RAG_Techniques`](https://github.com/NirDiamant/RAG_Techniques/blob/main/evaluation/evalute_rag.py) | `evaluation/evalute_rag.py` | GEval for "Correctness" (factual accuracy vs expected output) + Faithfulness + ContextualRelevancy across many RAG variants |
| [`kethlyncampos/DeepEval_101`](https://github.com/kethlyncampos/DeepEval_101) | `deep_eval.ipynb` | 5 custom GEval metrics (coherence, groundedness, relevance, accuracy, completeness) with Gemini 2.5 Flash as judge; radar-plot visualization |

---

## 3. Multi-Locale / Multilingual Evaluation

### Important: No Native i18n API in DeepEval

DeepEval has no `locale=` parameter, no language-detection layer, and no built-in non-English metric templates as of mid-2026. This is an acknowledged gap — see [Issue #718](https://github.com/confident-ai/deepeval/issues/718) ("Do you support evaluation on multi-lingual model and dataset?").

### The Community Pattern: G-Eval in the Target Language

Since G-Eval sends your `criteria` string directly to the judge LLM, you can write criteria in any language — and if you use a multilingual judge (GPT-4o, Gemini 2.5, or a local multilingual model), the evaluation works across locales:

```python
# Example: evaluating a French-language LLM response
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

relevance_fr = GEval(
    name="Pertinence",
    criteria="Évaluez si la réponse fournie répond directement à la question posée en français.",
    evaluation_steps=[
        "Vérifiez que la réponse est en français",
        "Vérifiez que la réponse aborde le sujet principal de la question",
        "Vérifiez l'absence d'informations hors sujet",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gemini-2.5-flash",   # multilingual judge
    threshold=0.7,
)

test_cases = [
    LLMTestCase(input="Quelle est la capitale de la France?", actual_output=fr_response),
    LLMTestCase(input="¿Cuál es la capital de España?", actual_output=es_response),
    LLMTestCase(input="日本の首都はどこですか？", actual_output=ja_response),
]
```

### Key GitHub Resources — Multi-Locale

| Repo | File/Link | What it does |
|------|-----------|--------------|
| [`confident-ai/deepeval` Issue #718](https://github.com/confident-ai/deepeval/issues/718) | GitHub Issue | Primary community discussion on multilingual support. Confirms no native API; G-Eval custom criteria is the workaround. |
| [`confident-ai/deepeval`](https://github.com/confident-ai/deepeval/blob/main/deepeval/metrics/g_eval/template.py) | `g_eval/template.py` | The prompt template that injects `criteria` into the LLM judge — this is the extension point for multilingual evaluation. |
| [`kethlyncampos/DeepEval_101`](https://github.com/kethlyncampos/DeepEval_101) | `deep_eval.ipynb` | Uses Gemini 2.5 Flash (natively multilingual) as judge — the exact pattern needed for locale-aware evaluation. |
| [`royapakzad/multilingual-ai-safety-evaluation`](https://github.com/royapakzad/multilingual-ai-safety-evaluation) | Full repo | Complementary project: evaluates LLMs across non-English languages using a human-rights-based rubric. Side-by-side English + target-language response comparison. Not DeepEval-native but directly composable. |
| [`codemaker2015/deepeval-experiments`](https://github.com/codemaker2015/deepeval-experiments) | Full repo | General scaffold (G-Eval, custom metrics, dataset construction) that is the community template for extending to non-English locales. |

### Designing a Multi-Locale Rubric Pipeline

```python
from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics.g_eval import Rubric

LOCALES = {
    "en": {"name": "Accuracy", "criteria": "Is the response factually accurate?"},
    "fr": {"name": "Exactitude", "criteria": "La réponse est-elle factuellement exacte?"},
    "es": {"name": "Precisión", "criteria": "¿Es la respuesta factualmente precisa?"},
    "ja": {"name": "正確性", "criteria": "回答は事実として正確ですか？"},
}

def make_locale_metric(locale: str) -> GEval:
    cfg = LOCALES[locale]
    return GEval(
        name=cfg["name"],
        criteria=cfg["criteria"],
        rubric=[
            Rubric(score_range=(0, 3),  expected_outcome="Inaccurate or misleading"),
            Rubric(score_range=(4, 6),  expected_outcome="Partially accurate"),
            Rubric(score_range=(7, 10), expected_outcome="Fully accurate"),
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model="gemini-2.5-flash",
        threshold=0.7,
    )

# Build test suite across locales
test_cases = []
for locale, question, response in locale_responses:
    tc = LLMTestCase(input=question, actual_output=response)
    tc.additional_metadata = {"locale": locale}
    test_cases.append((tc, make_locale_metric(locale)))

# Evaluate
for tc, metric in test_cases:
    metric.measure(tc)
    print(f"[{tc.additional_metadata['locale']}] score={metric.score:.2f} pass={metric.is_successful()}")
```

---

## Quick Reference: Top 10 DeepEval Repos

| # | Repo | Stars | Primary Use |
|---|------|-------|-------------|
| 1 | [`confident-ai/deepeval`](https://github.com/confident-ai/deepeval) | 8k+ | Official framework: all metrics, benchmarks, tracing |
| 2 | [`NirDiamant/RAG_Techniques`](https://github.com/NirDiamant/RAG_Techniques/blob/main/evaluation/evalute_rag.py) | 15k+ | GEval correctness + RAG metrics across many retrieval strategies |
| 3 | [`weaviate/recipes`](https://github.com/weaviate/recipes/blob/main/integrations/operations/deepeval/rag_evaluation_deepeval.ipynb) | 1k+ | DeepEval + Weaviate vector-DB RAG eval notebook |
| 4 | [`meteatamel/genai-beyond-basics`](https://github.com/meteatamel/genai-beyond-basics/blob/main/samples/evaluation/deepeval/rag_eval/test_rag_triad.py) | — | RAG Triad with Vertex AI + Gemini; CI/CD integration pattern |
| 5 | [`codemaker2015/deepeval-experiments`](https://github.com/codemaker2015/deepeval-experiments) | — | 12 experiments: all major metrics + tracing + MMLU benchmark |
| 6 | [`patchy631/ai-engineering-hub`](https://github.com/patchy631/ai-engineering-hub/blob/main/sonnet4-vs-o4/code_evaluation.py) | 9k+ | Multi-band Rubric GEval for model comparison (Sonnet vs o4) |
| 7 | [`kethlyncampos/DeepEval_101`](https://github.com/kethlyncampos/DeepEval_101) | — | 5 custom GEval metrics with Gemini; multilingual-ready judge |
| 8 | [`topoteretes/cognee`](https://github.com/topoteretes/cognee/blob/main/cognee/eval_framework/evaluation/metrics/rubric.py) | 3k+ | Binary per-criterion RubricMetric (YES/NO, fraction score) |
| 9 | [`vudayani/DeepEvalTests`](https://github.com/vudayani/DeepEvalTests) | — | G-Eval + AnswerRelevancy behind a REST `/evaluate` endpoint |
| 10 | [`LordZerror/GroundedDialogue`](https://github.com/LordZerror/GroundedDialogue/blob/main/eval/run_geval.py) | — | G-Eval rubric pipeline (no DeepEval runner) — Mode, Leakage, Format criteria |

---

## Latency + Rubric + Multi-Locale: Combined Pattern

```python
import time
from deepeval import evaluate
from deepeval.metrics import GEval, AnswerRelevancyMetric
from deepeval.metrics.g_eval import Rubric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

def evaluate_response(question: str, answer: str, locale: str = "en") -> dict:
    start = time.time()
    elapsed = time.time() - start   # measure real LLM latency externally

    tc = LLMTestCase(
        input=question,
        actual_output=answer,
        completion_time=elapsed,    # logged to Confident AI for latency tracking
    )

    quality = GEval(
        name="Quality",
        criteria=f"Is this response accurate, helpful, and appropriate for a {locale} speaker?",
        rubric=[
            Rubric(score_range=(0, 3),  expected_outcome="Poor — inaccurate or unhelpful"),
            Rubric(score_range=(4, 6),  expected_outcome="Adequate — partially useful"),
            Rubric(score_range=(7, 10), expected_outcome="Excellent — accurate and helpful"),
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model="gemini-2.5-flash",
        threshold=0.7,
    )
    relevancy = AnswerRelevancyMetric(threshold=0.7)

    evaluate([tc], [quality, relevancy])

    return {
        "locale": locale,
        "latency_s": elapsed,
        "quality_score": quality.score,
        "relevancy_score": relevancy.score,
        "passed": quality.is_successful() and relevancy.is_successful(),
    }
```

---

*Sources: [`confident-ai/deepeval`](https://github.com/confident-ai/deepeval) · [`NirDiamant/RAG_Techniques`](https://github.com/NirDiamant/RAG_Techniques) · [`weaviate/recipes`](https://github.com/weaviate/recipes) · [`meteatamel/genai-beyond-basics`](https://github.com/meteatamel/genai-beyond-basics) · [`patchy631/ai-engineering-hub`](https://github.com/patchy631/ai-engineering-hub) · [`topoteretes/cognee`](https://github.com/topoteretes/cognee) · [`codemaker2015/deepeval-experiments`](https://github.com/codemaker2015/deepeval-experiments) · [`kethlyncampos/DeepEval_101`](https://github.com/kethlyncampos/DeepEval_101) · [`vudayani/DeepEvalTests`](https://github.com/vudayani/DeepEvalTests) · [`LordZerror/GroundedDialogue`](https://github.com/LordZerror/GroundedDialogue) · [DeepEval Issue #718 — multilingual](https://github.com/confident-ai/deepeval/issues/718)*
