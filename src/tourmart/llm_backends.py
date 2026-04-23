"""LLM backend abstraction.

Two backends:
- `MockLLM`: for unit tests. Either fixed string or a callable (system, user) → str.
- `VLLMBackend`: stub; populated when Server B is ready.
"""
from __future__ import annotations

from typing import Callable, Protocol, Union


class LLMBackend(Protocol):
    def generate(
        self, system: str, user: str, max_tokens: int = 2048,
        json_schema: dict | None = None,
    ) -> str: ...

    def generate_batch(
        self,
        prompts: list[tuple[str, str]],
        max_tokens: int = None,
        json_schemas: list[dict | None] | None = None,
    ) -> list[str]:
        """Batch variant: each prompt is (system, user). Returns responses in order.

        If `json_schemas` is provided it must match the length of `prompts` — one
        schema (or None) per prompt for vLLM guided-decoding.
        """
        ...


ResponseOrFn = Union[str, Callable[[str, str], str]]


class MockLLM:
    """Deterministic backend for local tests.

    `response` may be a plain string (returned every call) or a callable taking
    (system_prompt, user_prompt) and returning the mock response string.
    """

    def __init__(self, response: ResponseOrFn = '{"decision_table": [], "recommendations": []}'):
        self.response = response
        self.call_log: list[tuple[str, str]] = []

    def generate(
        self, system: str, user: str, max_tokens: int = 2048,
        json_schema: dict | None = None,
    ) -> str:
        self.call_log.append((system, user))
        if callable(self.response):
            return self.response(system, user)
        return self.response

    def generate_batch(
        self, prompts: list[tuple[str, str]], max_tokens: int = None,
        json_schemas: list[dict | None] | None = None,
    ) -> list[str]:
        return [self.generate(sys, usr) for sys, usr in prompts]


class VLLMBackend:
    """Real vLLM backend. Populate once Server B is provisioned.

    Expected init: `VLLMBackend(model_path="${TOURMART_ROOT}/models/Qwen2.5-14B-Int4",
                                tensor_parallel_size=2, dtype="auto")`.
    """

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        dtype: str = "bfloat16",
        max_model_len: int = 6144,
        gpu_memory_utilization: float = 0.85,
        trust_remote_code: bool = True,
        default_max_tokens: int = 1024,
        quantization: str | None = None,
        **kwargs,
    ):
        try:
            from vllm import LLM, SamplingParams  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "vLLM not installed. VLLMBackend is only for GPU-equipped servers."
            ) from e
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.trust_remote_code = trust_remote_code
        self.default_max_tokens = default_max_tokens
        self.quantization = quantization
        self._kwargs = kwargs
        self._llm = None
        self._cached_sp = None

    def _lazy_init(self):
        if self._llm is not None:
            return
        from vllm import LLM
        init_kwargs = dict(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            dtype=self.dtype,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=self.trust_remote_code,
        )
        if self.quantization is not None:
            init_kwargs["quantization"] = self.quantization
        init_kwargs.update(self._kwargs)
        self._llm = LLM(**init_kwargs)

    @staticmethod
    def _chat_template(system: str, user: str) -> str:
        """Qwen2.5 chat template — manual construction (no tokenizer dependency)."""
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def _make_sp(self, max_tokens: int, json_schema: dict | None = None):
        """Build SamplingParams with optional JSON-guided decoding.

        Handles the vLLM API rename (0.6: GuidedDecodingParams.guided_decoding;
        0.9+: StructuredOutputsParams.structured_outputs). If neither is
        available OR no schema provided, returns a plain SamplingParams.
        """
        from vllm import SamplingParams
        if json_schema is None:
            return SamplingParams(temperature=0.0, max_tokens=max_tokens)
        # Try new API first (vLLM 0.9+).
        try:
            from vllm.sampling_params import StructuredOutputsParams
            so = StructuredOutputsParams(json=json_schema)
            return SamplingParams(
                temperature=0.0, max_tokens=max_tokens, structured_outputs=so,
            )
        except (ImportError, TypeError):
            pass
        # Fallback to older API.
        try:
            from vllm.sampling_params import GuidedDecodingParams
            gd = GuidedDecodingParams(json=json_schema)
            return SamplingParams(
                temperature=0.0, max_tokens=max_tokens, guided_decoding=gd,
            )
        except (ImportError, TypeError):
            import warnings
            warnings.warn(
                "vLLM structured-output / guided-decoding API unavailable; "
                "proceeding without schema enforcement.",
                RuntimeWarning,
            )
            return SamplingParams(temperature=0.0, max_tokens=max_tokens)

    def generate(
        self, system: str, user: str, max_tokens: int = None,
        json_schema: dict | None = None,
    ) -> str:
        self._lazy_init()
        mt = max_tokens if max_tokens is not None else self.default_max_tokens
        sp = self._make_sp(mt, json_schema)
        outputs = self._llm.generate([self._chat_template(system, user)], sp)
        return outputs[0].outputs[0].text

    def generate_batch(
        self,
        prompts: list[tuple[str, str]],
        max_tokens: int = None,
        json_schemas: list[dict | None] | None = None,
    ) -> list[str]:
        """Batched generation. Per-prompt JSON schemas pass through as
        per-request SamplingParams."""
        self._lazy_init()
        mt = max_tokens if max_tokens is not None else self.default_max_tokens
        rendered = [self._chat_template(s, u) for s, u in prompts]
        if json_schemas is None:
            sp_or_list = self._make_sp(mt, None)
        else:
            if len(json_schemas) != len(prompts):
                raise ValueError(
                    f"json_schemas length {len(json_schemas)} != prompts length {len(prompts)}"
                )
            sp_or_list = [self._make_sp(mt, s) for s in json_schemas]
        outputs = self._llm.generate(rendered, sp_or_list)
        return [o.outputs[0].text for o in outputs]


__all__ = ["LLMBackend", "MockLLM", "VLLMBackend"]
