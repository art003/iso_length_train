from __future__ import annotations

import io
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from prompts import REVIEW_SYSTEM, SYSTEM_PROMPT, review_prompt, user_prompt

JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.S)


def _compact_image(png: bytes, max_side: int = 1400, as_jpeg: bool = True) -> tuple[bytes, str]:
    from PIL import Image

    im = Image.open(io.BytesIO(png)).convert("RGB")
    w, h = im.size
    scale = max_side / max(w, h)
    if scale < 1:
        im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    if as_jpeg:
        im.save(buf, format="JPEG", quality=72, optimize=True)
        return buf.getvalue(), "image/jpeg"
    im.save(buf, format="PNG", optimize=True)
    return buf.getvalue(), "image/png"


@dataclass
class LlmUsage:
    provider: str
    model: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed_s: float = 0.0
    retries: int = 0


def _parse_json(text: str) -> dict:
    text = text.strip()
    m = JSON_FENCE.search(text)
    if m:
        text = m.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


class LlmClient:
    def __init__(self) -> None:
        self.provider = os.getenv("ISO_LLM_PROVIDER", "gemini").strip().lower()
        if self.provider == "openai":
            self.model = os.getenv("ISO_LLM_MODEL", "gpt-4o")
            self.api_key = os.getenv("OPENAI_API_KEY", "")
            self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
            if not self.api_key:
                raise RuntimeError("Нет OPENAI_API_KEY. Скопируйте .env.example в .env и укажите ключ.")
        elif self.provider == "openrouter":
            self.model = os.getenv("ISO_LLM_MODEL", "qwen/qwen3.8-27b:free")
            self.api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY", "")
            self.base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
            if not self.api_key:
                raise RuntimeError("Нет OPENROUTER_API_KEY. Ключ: https://openrouter.ai/keys")
        elif self.provider == "codex":
            self.model = os.getenv("ISO_LLM_MODEL", "gpt-5-codex")
            self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
            self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
            if not self.api_key:
                raise RuntimeError(
                    "Нет OPENAI_API_KEY для Codex. Ключ: https://platform.openai.com/api-keys"
                )
        elif self.provider == "groq":
            self.model = os.getenv("ISO_LLM_MODEL", "qwen/qwen3.8-27b")
            self.api_key = os.getenv("GROQ_API_KEY", "").strip()
            self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
            if not self.api_key:
                raise RuntimeError(
                    "Нет GROQ_API_KEY. Бесплатный ключ без карты: https://console.groq.com/keys "
                    "Потом в .env строка GROQ_API_KEY=gsk_..."
                )
        elif self.provider == "gemini":
            self.model = os.getenv("ISO_LLM_MODEL", "gemini-2.5-flash")
            self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
            if not self.api_key:
                raise RuntimeError("Нет GEMINI_API_KEY. Скопируйте .env.example в .env и укажите ключ.")
        elif self.provider == "ollama":
            self.model = os.getenv("ISO_LLM_MODEL", "qwen2.5vl")
            self.base_url = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
            self.api_key = ""
        elif self.provider == "replay":
            self.model = os.getenv("ISO_LLM_MODEL", "gemini-2.5-flash")
            self.replay_dir = Path(os.getenv("ISO_REPLAY_DIR", "output/llm_raw"))
            self.api_key = ""
        elif self.provider == "none":
            self.model = os.getenv("ISO_LLM_MODEL", "local")
            self.api_key = ""
        else:
            raise RuntimeError(f"Неизвестный ISO_LLM_PROVIDER={self.provider}")
        self.usage = LlmUsage(provider=self.provider, model=self.model)

    def classify_sheet(
        self,
        sheet_no: int,
        line_id: str,
        candidates: list[dict],
        image_png: bytes,
        first: dict | None = None,
        hint: str = "",
    ) -> dict:
        wanted = {c["id"] for c in candidates}
        if first:
            prompt = review_prompt(sheet_no, line_id, candidates, first, hint)
            system = REVIEW_SYSTEM
        else:
            prompt = user_prompt(sheet_no, line_id, candidates)
            system = SYSTEM_PROMPT
        last_err = None
        tries = 2 if self.provider in ("openrouter", "groq") else 3
        for attempt in range(tries):
            if attempt:
                self.usage.retries += 1
                if self.provider == "groq":
                    time.sleep(int(os.getenv("GROQ_PAUSE_S", "70")))
                elif self.provider == "openrouter":
                    time.sleep(10)
                prompt = prompt + "\nПовтори JSON целиком. Нужны ВСЕ id: " + ", ".join(sorted(wanted))
            raw = self._complete(prompt, image_png, sheet_no, system)
            try:
                data = _parse_json(raw)
            except json.JSONDecodeError as e:
                last_err = e
                continue
            items = data.get("items") or []
            got = {it.get("id") for it in items}
            if wanted <= got:
                return data
            missing = sorted(wanted - got)
            last_err = RuntimeError(f"не хватает id: {missing}")
            prompt = prompt + f"\nНе хватает меток: {missing}. Верни полный JSON."
        raise RuntimeError(f"LLM не вернул полный JSON: {last_err}")

    def _complete(
        self,
        prompt: str,
        image_png: bytes,
        sheet_no: int | None = None,
        system: str | None = None,
    ) -> str:
        t0 = time.perf_counter()
        sys_p = system or SYSTEM_PROMPT
        if self.provider in ("openai", "openrouter", "groq"):
            text, inn, out = self._openai(prompt, image_png, sys_p)
        elif self.provider == "codex":
            text, inn, out = self._codex(prompt, image_png, sys_p)
        elif self.provider == "ollama":
            text, inn, out = self._ollama(prompt, image_png, sys_p)
        elif self.provider == "replay":
            text, inn, out = self._replay(sheet_no)
            self.usage.elapsed_s += time.perf_counter() - t0
            return text
        else:
            text, inn, out = self._gemini(prompt, image_png, sys_p)
        self.usage.calls += 1
        self.usage.input_tokens += inn
        self.usage.output_tokens += out
        self.usage.elapsed_s += time.perf_counter() - t0
        return text

    def _post_json(self, url: str, headers: dict | None, payload: dict) -> dict:
        wait = 70.0 if self.provider == "groq" else 25.0
        last = None
        tries = 5 if self.provider == "groq" else 3
        for attempt in range(tries):
            with httpx.Client(timeout=180.0) as client:
                r = client.post(url, headers=headers, json=payload)
            if r.status_code == 401:
                if self.provider == "groq":
                    raise RuntimeError(
                        "Groq не принял ключ (401). В .env нужен GROQ_API_KEY с "
                        "https://console.groq.com/keys , строка начинается с gsk_"
                    )
                raise RuntimeError(
                    "OpenAI не принял ключ (401). Нужен ключ с platform.openai.com, "
                    "OPENAI_API_KEY=sk-... без кавычек."
                )
            if r.status_code == 403:
                snippet = (r.text or "")[:280].replace("\n", " ")
                if self.provider == "groq":
                    raise RuntimeError(
                        "Groq отказал (403). Ключ есть, но модель/тариф не пускает. "
                        "Проверь лимиты в console.groq.com/settings/limits. "
                        f"{snippet}"
                    )
                r.raise_for_status()
            if r.status_code in (400, 404):
                snippet = (r.text or "")[:400].replace("\n", " ")
                if self.provider == "groq":
                    raise RuntimeError(
                        f"Groq {r.status_code}: модель {self.model} не приняла запрос. "
                        "На бесплатном ключе сейчас жива qwen/qwen3.8-27b. "
                        f"{snippet}"
                    )
                r.raise_for_status()
            if r.status_code != 429:
                r.raise_for_status()
                return r.json()
            body = (r.text or "").lower()
            if "insufficient_quota" in body:
                raise RuntimeError(
                    "Ключ ок, но на OpenAI нет денег/квоты. Пополни: "
                    "https://platform.openai.com/settings/organization/billing"
                )
            if "free-models-per-day" in body:
                raise RuntimeError("Дневной лимит бесплатной модели. Подожди несколько часов или смени модель.")
            if "otpm" in body or "output tokens per minute" in body:
                wait = max(wait, 70.0)
            hdr = r.headers.get("retry-after") or r.headers.get("Retry-After")
            try:
                wait = max(wait, float(hdr)) if hdr else wait
            except ValueError:
                pass
            snippet = (r.text or "")[:240].replace("\n", " ")
            print(f"лимит API (429), пауза {int(wait)} с ({attempt + 1}/{tries}). {snippet}", flush=True)
            self.usage.retries += 1
            time.sleep(wait)
            wait = min(wait * 2, 90.0)
            last = r
        extra = ""
        if last is not None:
            extra = " Ответ: " + (last.text or "")[:240].replace("\n", " ")
        raise RuntimeError(
            "Бесплатный лимит этой модели кончился (429)." + extra
            + " Подожди час или смени модель в .env."
        )

    def _replay(self, sheet_no: int | None) -> tuple[str, int, int]:
        if sheet_no is None:
            raise RuntimeError("replay: не передан номер листа")
        path = Path(self.replay_dir) / f"sheet_{int(sheet_no):02d}.json"
        if not path.exists():
            raise RuntimeError(f"replay: нет файла {path}")
        return path.read_text(encoding="utf-8"), 0, 0

    def _ollama(self, prompt: str, image_png: bytes, system: str = SYSTEM_PROMPT) -> tuple[str, int, int]:
        import base64

        side = int(os.getenv("OLLAMA_IMAGE_SIDE", "2000"))
        img, _mime = _compact_image(image_png, max_side=side, as_jpeg=False)
        b64 = base64.b64encode(img).decode("ascii")
        options = {
            "temperature": float(os.getenv("OLLAMA_TEMPERATURE", "0")),
            "top_p": float(os.getenv("OLLAMA_TOP_P", "0.1")),
            "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "16384")),
            "num_predict": int(os.getenv("OLLAMA_NUM_PREDICT", "2048")),
            "repeat_penalty": float(os.getenv("OLLAMA_REPEAT_PENALTY", "1.05")),
            "seed": int(os.getenv("OLLAMA_SEED", "42")),
        }
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "keep_alive": "30m",
            "options": options,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt, "images": [b64]},
            ],
        }
        try:
            with httpx.Client(timeout=600.0) as client:
                r = client.post(f"{self.base_url}/api/chat", json=payload)
        except httpx.ConnectError as e:
            raise RuntimeError(
                "Ollama не запущена. Поставь https://ollama.com/download , "
                f"потом в терминале: ollama pull {self.model}"
            ) from e
        if r.status_code == 404:
            raise RuntimeError(f"Нет модели {self.model}. В терминале: ollama pull {self.model}")
        r.raise_for_status()
        data = r.json()
        text = ((data.get("message") or {}).get("content")) or ""
        return text, 0, 0

    def _openai(self, prompt: str, image_png: bytes, system: str = SYSTEM_PROMPT) -> tuple[str, int, int]:
        import base64

        if self.provider == "groq":
            side = int(os.getenv("GROQ_IMAGE_SIDE", "1800"))
            img_bytes, mime = _compact_image(image_png, max_side=side, as_jpeg=False)
        else:
            img_bytes, mime = _compact_image(image_png)
        b64 = base64.b64encode(img_bytes).decode("ascii")
        image = {"url": f"data:{mime};base64,{b64}"}
        if self.provider not in ("openrouter", "groq"):
            image["detail"] = "high"
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": image},
                    ],
                },
            ],
        }
        if self.provider != "openrouter":
            payload["response_format"] = {"type": "json_object"}
        if self.provider == "groq":
            # groq бесплатно даёт 1000 токенов выхода в минуту
            payload["max_completion_tokens"] = min(int(os.getenv("GROQ_MAX_TOKENS", "900")), 900)
            if self.model.startswith("qwen/"):
                payload["reasoning_effort"] = os.getenv("GROQ_REASONING", "none")
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/art003/iso-length"
            headers["X-Title"] = "iso-length"
        data = self._post_json(
            f"{self.base_url}/chat/completions",
            headers=headers,
            payload=payload,
        )
        text = data["choices"][0]["message"].get("content") or ""
        usage = data.get("usage") or {}
        return text, int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)

    def _codex(self, prompt: str, image_png: bytes, system: str = SYSTEM_PROMPT) -> tuple[str, int, int]:
        import base64

        img_bytes, mime = _compact_image(image_png, max_side=1800, as_jpeg=False)
        b64 = base64.b64encode(img_bytes).decode("ascii")
        payload = {
            "model": self.model,
            "instructions": system,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": f"data:{mime};base64,{b64}"},
                    ],
                }
            ],
            "max_output_tokens": int(os.getenv("CODEX_MAX_TOKENS", "4096")),
            "text": {"format": {"type": "json_object"}},
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = self._post_json(f"{self.base_url}/responses", headers, payload)
        text = (data.get("output_text") or "").strip()
        if not text:
            chunks = []
            for item in data.get("output") or []:
                for part in item.get("content") or []:
                    if part.get("text"):
                        chunks.append(part["text"])
            text = "\n".join(chunks)
        usage = data.get("usage") or {}
        return text, int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)

    def _gemini(self, prompt: str, image_png: bytes, system: str = SYSTEM_PROMPT) -> tuple[str, int, int]:
        import base64

        b64 = base64.b64encode(image_png).decode("ascii")
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
            f"?key={self.api_key}"
        )
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {"inlineData": {"mimeType": "image/png", "data": b64}},
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
            },
        }
        data = self._post_json(url, headers=None, payload=payload)
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        usage = data.get("usageMetadata") or {}
        return text, int(usage.get("promptTokenCount") or 0), int(usage.get("candidatesTokenCount") or 0)
