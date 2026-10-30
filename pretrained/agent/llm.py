"""One client for every LLM call our agent makes (local Ollama, OpenAI-compatible API).

Records every request/response pair so runs can be audited and turned into training data.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from openai import OpenAI

from common.paths import LLAMA_MODEL_NAME, OLLAMA_BASE_URL

DEFAULT_TEMPERATURE = 0.2          # same as RESTestBench


@dataclass
class LLM:
    model: str = LLAMA_MODEL_NAME
    base_url: str = OLLAMA_BASE_URL
    temperature: float = DEFAULT_TEMPERATURE
    max_retries: int = 5
    calls: list = field(default_factory=list)

    def __post_init__(self):
        self.client = OpenAI(base_url=self.base_url, api_key="ollama", timeout=600)

    def chat(self, messages: list[dict], max_tokens: int = 2048, tag: str = "",
             temperature_override: float | None = None, **kw) -> str:
        last_err = None
        for attempt in range(self.max_retries):
            try:
                t0 = time.time()
                r = self.client.chat.completions.create(
                    model=self.model, messages=messages,
                    temperature=self.temperature if temperature_override is None else temperature_override,
                    max_tokens=max_tokens, **kw)
                text = r.choices[0].message.content or ""
                self.calls.append({
                    "tag": tag, "messages": messages, "response": text,
                    "temperature": self.temperature if temperature_override is None else temperature_override,
                    "prompt_tokens": getattr(r.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(r.usage, "completion_tokens", None),
                    "seconds": round(time.time() - t0, 2)})
                return text
            except Exception as e:  # server restart, timeout: retry
                last_err = e
                time.sleep(3 * (attempt + 1))
        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts: {last_err}")

    def chat_tools(self, messages: list[dict], tools: list[dict], max_tokens: int = 2048,
                   tag: str = "") -> dict:
        """One tool-calling turn. Returns {"content": str, "tool_calls": [{"id", "name", "arguments"}]}."""
        last_err = None
        for attempt in range(self.max_retries):
            try:
                t0 = time.time()
                r = self.client.chat.completions.create(model=self.model, messages=messages, tools=tools,
                                                        temperature=self.temperature, max_tokens=max_tokens)
                m = r.choices[0].message
                calls = [{"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
                         for tc in (m.tool_calls or [])]
                out = {"content": m.content or "", "tool_calls": calls}
                self.calls.append({
                    "tag": tag, "messages": messages, "response": out, "temperature": self.temperature,
                    "prompt_tokens": getattr(r.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(r.usage, "completion_tokens", None),
                    "seconds": round(time.time() - t0, 2)})
                return out
            except Exception as e:
                last_err = e
                time.sleep(3 * (attempt + 1))
        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts: {last_err}")

    def complete(self, prompt: str, max_tokens: int = 2048, tag: str = "",
                 temperature_override: float | None = None, **kw) -> str:
        return self.chat([{"role": "user", "content": prompt}], max_tokens=max_tokens, tag=tag,
                         temperature_override=temperature_override, **kw)

    def usage(self) -> dict:
        return {"requests": len(self.calls),
                "prompt_tokens": sum(c["prompt_tokens"] or 0 for c in self.calls),
                "completion_tokens": sum(c["completion_tokens"] or 0 for c in self.calls)}
