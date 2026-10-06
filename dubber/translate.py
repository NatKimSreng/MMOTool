"""
Translation to spoken Khmer: prompts, Gemini / OpenAI / Groq LLMs and free Google fallback.
"""
import threading
import time
from .core import (
    load_config,
)
from .asr import (
    _has_khmer,
)


def _parse_numbered_translations(raw, expected_count):
    """Parse '1. text' style LLM output into a list of strings.
    Matches by line number first, so a skipped/merged line can't shift every line after it."""
    import re
    lines = [ln.strip() for ln in (raw or "").split("\n") if ln.strip()]
    by_num = {}
    for ln in lines:
        m = re.match(r"^\s*(\d{1,3})\s*[.)\-:]\s*(.*)$", ln)
        if m:
            n = int(m.group(1))
            txt = m.group(2).strip()
            if len(txt) >= 2 and txt[0] in "\"'“”«»" and txt[-1] in "\"'“”«»":
                txt = txt[1:-1].strip()
            if 1 <= n <= expected_count and txt and n not in by_num:
                by_num[n] = txt
    if len(by_num) >= max(1, expected_count // 2):
        return [by_num.get(i + 1, "") for i in range(expected_count)]

    mapped = []
    for ln in lines:
        cleaned = ln
        if len(cleaned) > 2 and cleaned[0].isdigit():
            i = 0
            while i < len(cleaned) and (cleaned[i].isdigit() or cleaned[i] in ".)-: "):
                i += 1
            if i > 0 and i < len(cleaned):
                cleaned = cleaned[i:].strip()
        # strip accidental quotes models sometimes wrap around the line
        if len(cleaned) >= 2 and cleaned[0] in "\"'“”«»" and cleaned[-1] in "\"'“”«»":
            cleaned = cleaned[1:-1].strip()
        if cleaned:
            mapped.append(cleaned)
    while len(mapped) < expected_count:
        mapped.append("")
    return mapped[:expected_count]


def _polish_khmer_line(text: str) -> str:
    """Cleanup so TTS and reading feel natural (spoken Khmer)."""
    if not text:
        return ""
    t = str(text).strip()
    while "  " in t:
        t = t.replace("  ", " ")
    # common machine-translation stiffness / punctuation fixes
    for bad, good in (
        (" .", "."),
        (" ,", ","),
        (" !", "!"),
        (" ?", "?"),
        ("。。", "។"),
        ("...", "…"),
        ("..", "។"),
        ("，", "،"),
        ("។។", "។"),
        (" ។", "។"),
        ("។ ", "។ "),
        # stiff formal → more spoken (light, safe replacements)
        ("ខ្ញុំនឹង", "ខ្ញុំនឹង"),  # keep
        ("សូមអរគុណច្រើនណាស់", "អរគុណច្រើន"),
        ("តើអ្នកចង់", "តើអ្នកចង់"),
    ):
        t = t.replace(bad, good)
    import re
    # doubled end marks ("។." "?។" "!.") make the voice stop twice
    t = re.sub(r"([។!?…])[\.។]+", r"\1", t)
    t = re.sub(r"\.+([។!?])", r"\1", t)
    # a Latin full stop after Khmer words → Khmer full stop (read as a proper sentence end)
    if _has_khmer(t):
        t = re.sub(r"(?<=[ក-៿​])\.(\s|$)", r"។\1", t)
    # strip leading/trailing quotes that models sometimes add
    if len(t) >= 2 and t[0] in "\"'“”«»" and t[-1] in "\"'“”«»":
        t = t[1:-1].strip()
    return t.strip()


# Specialized Khmer Translation Prompts tailored for Movie Recaps & Dramas (DAI Dubber style)
PROMPT_STYLES = {
    "recap": (
        "You are an expert Cambodian movie recap writer and voice-over dialogue artist (អ្នកនិពន្ធសម្រាយរឿង និងបញ្ចូលសំឡេងភាពយន្តខ្មែរ).\n"
        "Your task: Turn each line into exciting, natural, dramatic storytelling spoken Khmer specifically crafted for movie recaps (សម្រាយរឿង / និទានរឿង).\n"
        "\n"
        "PRIORITY: THRILLING STORYTELLING FLOW\n"
        "- Sound like a skilled narrator talking directly to viewers on Facebook/YouTube/TikTok.\n"
        "- Use narrative cadence and transitions naturally (ឧ. 'ពេលនោះឯង...', 'ស្រាប់តែ...', 'មិនបង្អង់យូរ...', 'បន្ទាប់មកទៀត...', 'គាត់ក៏...').\n"
        "- Keep sentences punchy, rhythmic, and natural to listen to via text-to-speech.\n"
        "- Maintain speaker energy and emotion across line boundaries.\n"
        "\n"
        "HARD RULES:\n"
        "1. Output ONLY Khmer Unicode (អក្សរខ្មែរ). No English/Chinese/Latin except essential names or numbers.\n"
        "2. Meaning must stay accurate — do not drop or invent plot facts.\n"
        "3. Everyday spoken storytelling Khmer only (ទេ, ណា, ហ្នឹង, ម៉េច, តើ, ទៅ, ចុះ).\n"
        "4. Reply with the exact same line numbers. No intro, no commentary.\n"
        "\n"
        "Format exactly:\n"
        "1. <khmer>\n"
        "2. <khmer>"
    ),
    "dialogue": (
        "You are an elite Cambodian dialogue director and script translator for movies and dramas (អ្នកដឹកនាំបញ្ចូលសំឡេងភាពយន្តខ្មែរ).\n"
        "Your task: Translate each line into authentic, emotional, conversational spoken Khmer (សន្ទនាតួអង្គពិតៗដូចរឿងភាគ).\n"
        "\n"
        "PRIORITY: REALISTIC CHARACTER DIALOGUE\n"
        "- Sound like real actors speaking in a high-end dubbed film.\n"
        "- Use proper Khmer pronouns based on character relationships (បង/អូន, ឯង/ខ្ញុំ, លោក/នាង, ពួកយើង, គាត់).\n"
        "- Use conversational particles and question tags naturally (ទេ, ណា, ហ្នឹង, ម៉េច, តើ, ទៅ, ចុះ, លេង).\n"
        "- Short, spoken rhythm (one idea per line). Avoid long bookish clauses.\n"
        "\n"
        "HARD RULES:\n"
        "1. Output ONLY Khmer Unicode (អក្សរខ្មែរ).\n"
        "2. Keep pronouns and tone consistent with previous dialogue.\n"
        "3. Reply with the exact same line numbers only.\n"
        "\n"
        "Format exactly:\n"
        "1. <khmer>\n"
        "2. <khmer>"
    ),
    "general": (
        "You are a native Khmer subtitle translator.\n"
        "Your task: Translate each line into clear, natural spoken Khmer for video subtitles and voice-over.\n"
        "RULES:\n"
        "1. Output ONLY Khmer Unicode (អក្សរខ្មែរ).\n"
        "2. Clear, simple, easy to read and understand.\n"
        "3. Reply with the exact same line numbers only.\n"
        "\n"
        "Format exactly:\n"
        "1. <khmer>\n"
        "2. <khmer>"
    )
}

_KHMER_SUB_SYSTEM = PROMPT_STYLES["recap"]


def _apply_glossary_replacement(text: str, glossary: dict = None) -> str:
    """Deterministic post-translation replacement of names and terms defined in glossary."""
    if not text:
        return text
    if glossary is None:
        cfg = load_config()
        glossary = cfg.get("glossary", {})
    if isinstance(glossary, dict):
        for term, replacement in glossary.items():
            if term and replacement and term in text:
                text = text.replace(term, replacement)
    return text


def _build_translate_user_prompt(batch, prev_line_khmer="", style="recap", prev_source_lines=None):
    """Numbered source lines + previous Khmer line for continuous spoken flow."""
    numbered = "\n".join(f"{i+1}. {s['text']}" for i, s in enumerate(batch))
    ctx = ""
    if prev_source_lines:
        ctx += (
            "Lines spoken just before this batch (context only — do NOT translate them):\n"
            + "\n".join(prev_source_lines)
            + "\n\n"
        )
    if prev_line_khmer and _has_khmer(prev_line_khmer):
        ctx += (
            "Previous Khmer line (for flow and pronoun continuity only — do NOT repeat it):\n"
            f"{prev_line_khmer}\n\n"
            "Continue in the same style and relationship tone.\n\n"
        )

    cfg = load_config()
    glossary = cfg.get("glossary", {})
    glossary_lines = []
    if isinstance(glossary, dict) and glossary:
        for k, v in glossary.items():
            if k and v:
                glossary_lines.append(f"- {k} -> {v}")
    glossary_ctx = ""
    if glossary_lines:
        glossary_ctx = (
            "GLOSSARY & CHARACTER NAMES (Translate EXACTLY as specified below):\n"
            + "\n".join(glossary_lines[:30])
            + "\n\n"
        )

    instruction = (
        "Translate each line below into natural spoken Khmer for movie recaps / subtitles.\n"
        if style == "recap" else
        "Translate each line below into authentic conversational Khmer dialogue.\n"
    )
    return (
        f"{ctx}"
        f"{glossary_ctx}"
        f"{instruction}"
        f"There are exactly {len(batch)} numbered lines. Output exactly {len(batch)} lines, "
        "each starting with its number (\"1. ...\"). Never merge, skip or split lines. "
        "No notes or explanations.\n"
        "One short, clear spoken line per number.\n"
        "Make the batch flow as continuous audio dialogue.\n\n"
        f"{numbered}"
    )


GROQ_CHAT_MODELS = ("openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b")

# Free Gemini keys get ~20 requests/day PER MODEL, so we rotate through several Flash
# models. A model that is out of quota (or retired) is skipped until its reset time
# instead of being retried — retrying a daily quota just wastes minutes.
GEMINI_MODELS = [
    "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.8-flash",
    "gemini-2.5-flash", "gemini-3.5-flash-lite", "gemini-2.5-flash-lite",
]
_GEMINI_BLOCKED = {}  # model -> unix time when it may be tried again
_GEMINI_LOCK = threading.Lock()


def _gemini_block(model, seconds):
    with _GEMINI_LOCK:
        _GEMINI_BLOCKED[model] = time.time() + seconds


def _gemini_retry_seconds(resp_json):
    """Seconds until a 429 resets, from the RetryInfo detail (default 60)."""
    try:
        for d in resp_json.get("error", {}).get("details", []):
            if "RetryInfo" in d.get("@type", ""):
                return float(str(d.get("retryDelay", "60s")).rstrip("s"))
    except Exception:
        pass
    return 60.0


def _gemini_chat_translate(system_prompt, user_prompt):
    """
    Translate with Google Gemini, rotating through Flash models (best Khmer quality).
    Returns (text, "gemini-<model>") or (None, last_error).
    """
    cfg = load_config()
    api_key = (cfg.get("GEMINI_API_KEY") or "").strip()
    if not api_key:
        return None, "GEMINI_API_KEY not configured"

    import requests
    models = list(dict.fromkeys(([cfg["gemini_model"]] if cfg.get("gemini_model") else []) + GEMINI_MODELS))
    last_err = None

    for _round in range(3):
        for model in models:
            if _GEMINI_BLOCKED.get(model, 0) > time.time():
                continue
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            gen_cfg = {"temperature": 0.25, "maxOutputTokens": 16384}
            if model.startswith("gemini-2.5"):
                # no hidden "thinking" — it eats output tokens and adds latency for plain translation
                gen_cfg["thinkingConfig"] = {"thinkingBudget": 0}
            payload = {
                "system_instruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                "generationConfig": gen_cfg,
            }
            for attempt in range(5):
                try:
                    r = requests.post(url, json=payload, timeout=120)
                except Exception as e:
                    last_err = f"Gemini {model}: {e}"
                    time.sleep(2)
                    continue
                if r.status_code == 200:
                    data = r.json()
                    cand = data.get("candidates", [])
                    if cand and "parts" in cand[0].get("content", {}):
                        text = "".join(p.get("text", "") for p in cand[0]["content"]["parts"] if not p.get("thought")).strip()
                        if text:
                            return text, f"gemini-{model}"
                    last_err = f"Gemini {model}: empty response"
                    break
                try:
                    body = r.json()
                except Exception:
                    body = {}
                last_err = f"Gemini {model} HTTP {r.status_code}: {r.text[:160]}"
                if r.status_code == 429:
                    wait = _gemini_retry_seconds(body)
                    if wait > 90 or "PerDay" in r.text:
                        print(f"[gemini] {model} out of daily quota — skipping it for {wait/3600:.1f}h")
                    # per-minute limit: rest this model and move straight to the next one
                    _gemini_block(model, wait + 1)
                    break
                if r.status_code in (500, 502, 503, 504):
                    time.sleep(3 * (attempt + 1))
                    continue
                if r.status_code in (400, 403, 404):
                    _gemini_block(model, 24 * 3600)  # retired / not allowed for this key
                break
        # every model is resting on its per-minute limit — wait for the first to free up
        soonest = min((_GEMINI_BLOCKED.get(m, 0) for m in models), default=0) - time.time()
        if soonest <= 0 or soonest > 65:
            break
        time.sleep(soonest + 0.5)
    return None, last_err


def _llm_chat_translate(messages, system_prompt=None, user_prompt=None, engine_choice="auto", style="recap"):
    """
    Call best available LLM for Khmer translation with priority:
    1. Gemini AI (if key present or selected) - Best Khmer fluency
    2. OpenAI (gpt-4o-mini / gpt-4o)
    3. Groq (llama-3.3-70b-versatile)
    """
    cfg = load_config()
    gemini_key = (cfg.get("GEMINI_API_KEY") or "").strip()
    openai_key = (cfg.get("OPENAI_API_KEY") or "").strip()
    groq_key = (cfg.get("GROQ_API_KEY") or "").strip()
    last_err = None

    sys_text = system_prompt or PROMPT_STYLES.get(style, PROMPT_STYLES["recap"])
    usr_text = user_prompt or (messages[-1]["content"] if messages else "")

    # --- 1) Google Gemini (Top quality Khmer) ---
    if (engine_choice in ("auto", "gemini")) and gemini_key:
        raw, model = _gemini_chat_translate(sys_text, usr_text)
        if raw:
            return raw, model
        last_err = model
        print(f"[Gemini translate fallback]: {last_err}")

    # --- 2) OpenAI ---
    if (engine_choice in ("auto", "openai") or (engine_choice == "gemini" and not gemini_key)) and openai_key:
        ai_messages = [
            {"role": "system", "content": sys_text},
            {"role": "user", "content": usr_text}
        ]
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            for model in ("gpt-4o-mini", "gpt-4o"):
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=ai_messages,
                        temperature=0.25,
                        max_tokens=4000,
                    )
                    raw = (resp.choices[0].message.content or "").strip()
                    if raw:
                        return raw, model
                except Exception as e:
                    last_err = e
                    continue
        except Exception:
            import requests
            headers = {"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"}
            for model in ("gpt-4o-mini", "gpt-4o"):
                try:
                    r = requests.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers=headers,
                        json={"model": model, "messages": ai_messages, "temperature": 0.25, "max_tokens": 4000},
                        timeout=45
                    )
                    if r.status_code == 200:
                        res_json = r.json()
                        raw = (res_json.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
                        if raw:
                            return raw, model
                    else:
                        last_err = f"OpenAI error {r.status_code}: {r.text[:120]}"
                except Exception as e:
                    last_err = e
                    continue

    # --- 3) Groq ---
    if (engine_choice in ("auto", "groq") or not (gemini_key or openai_key)) and groq_key:
        ai_messages = [
            {"role": "system", "content": sys_text},
            {"role": "user", "content": usr_text}
        ]
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            for model in GROQ_CHAT_MODELS:
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=ai_messages,
                        temperature=0.25,
                        max_tokens=16000,
                    )
                    raw = (resp.choices[0].message.content or "").strip()
                    if raw:
                        return raw, model
                except Exception as e:
                    last_err = e
                    continue
        except Exception:
            import requests
            headers = {"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"}
            for model in GROQ_CHAT_MODELS:
                try:
                    r = requests.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers=headers,
                        json={"model": model, "messages": ai_messages, "temperature": 0.25, "max_tokens": 4000},
                        timeout=45
                    )
                    if r.status_code == 200:
                        res_json = r.json()
                        raw = (res_json.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
                        if raw:
                            return raw, model
                    else:
                        last_err = f"Groq error {r.status_code}: {r.text[:120]}"
                except Exception as e:
                    last_err = e
                    continue

    return None, last_err


_GOOGLE_FREE_LOCK = threading.Lock()
_GOOGLE_FREE_LAST = [0.0]


def _google_free_throttle(min_gap=0.5):
    """Free Google endpoints block bursts — keep calls spaced out across all threads."""
    with _GOOGLE_FREE_LOCK:
        wait = _GOOGLE_FREE_LAST[0] + min_gap - time.time()
        if wait > 0:
            time.sleep(wait)
        _GOOGLE_FREE_LAST[0] = time.time()


def _translate_lines_google_batch(texts):
    """Translate many lines with ONE free Google request (lines joined by newlines).
    Falls back to slow single-line calls only if the line count doesn't match."""
    import requests
    texts = [(t or "").strip().replace("\n", " ") for t in texts]
    for attempt in range(4):
        _google_free_throttle()
        try:
            r = requests.post(
                "https://translate.googleapis.com/translate_a/single",
                params={"client": "gtx", "sl": "auto", "tl": "km", "dt": "t"},
                data={"q": "\n".join(texts)},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=40,
            )
            if r.status_code == 200:
                joined = "".join(item[0] for item in r.json()[0] if item and item[0])
                lines = [ln.strip() for ln in joined.split("\n")]
                if len(lines) == len(texts):
                    return [_polish_khmer_line(ln) if ln else t for ln, t in zip(lines, texts)]
                break
            if r.status_code == 429:
                time.sleep(10 * (attempt + 1))
                continue
            break
        except Exception as e:
            print(f"[google batch]: {e}")
            time.sleep(3)
    return [_translate_one_google(t) or t for t in texts]


def _translate_one_google(text, retries=3, context_prev="", context_next=""):
    """Free translate to Khmer. Tries multiple free backends so one blockage doesn't kill the job.
    Optional surrounding context helps coherence (we still return only the middle line).
    Order: deep-translator Google → direct Google gtx → MyMemory → clients5 Google."""
    import time
    text = (text or "").strip()
    if not text:
        return ""
    last_err = None

    # Build a short context block so Google gets dialogue flow (then extract middle)
    query = text
    if context_prev or context_next:
        parts = []
        if context_prev:
            parts.append(context_prev.strip())
        parts.append(f">>> {text} <<<")
        if context_next:
            parts.append(context_next.strip())
        query = "\n".join(parts)

    def _ok(kh):
        return bool(kh and str(kh).strip())

    def _extract_middle(kh: str) -> str:
        """If we sent context, try to pull only the target line."""
        if not (context_prev or context_next):
            return kh
        # Prefer content between markers if model echoed them
        if ">>>" in kh and "<<<" in kh:
            try:
                mid = kh.split(">>>", 1)[1].split("<<<", 1)[0].strip()
                if mid:
                    return mid
            except Exception:
                pass
        # Fallback: take the line that looks most like a single subtitle
        lines = [ln.strip() for ln in kh.replace(">>>", "").replace("<<<", "").split("\n") if ln.strip()]
        if len(lines) == 1:
            return lines[0]
        if len(lines) >= 3:
            return lines[1]  # middle of prev / target / next
        if lines:
            # pick the longest non-empty as best guess
            return max(lines, key=len)
        return kh

    for attempt in range(retries):
        _google_free_throttle()
        # 1) deep-translator Google
        try:
            from deep_translator import GoogleTranslator
            kh = GoogleTranslator(source="auto", target="km").translate(query)
            if _ok(kh):
                return _polish_khmer_line(_extract_middle(str(kh)))
        except Exception as e:
            last_err = e
            print(f"[translate deep_translator attempt {attempt+1}]: {e}")

        # 2) direct Google translate endpoint (no key)
        try:
            import requests
            r = requests.get(
                "https://translate.googleapis.com/translate_a/single",
                params={
                    "client": "gtx",
                    "sl": "auto",
                    "tl": "km",
                    "dt": "t",
                    "q": query,
                },
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=25,
            )
            if r.status_code == 200:
                data = r.json()
                parts = [item[0] for item in data[0] if item and item[0]]
                kh = "".join(parts).strip()
                if _ok(kh):
                    return _polish_khmer_line(_extract_middle(kh))
            else:
                last_err = f"HTTP {r.status_code}"
                print(f"[translate gtx attempt {attempt+1}]: HTTP {r.status_code}")
        except Exception as e:
            last_err = e
            print(f"[translate gtx attempt {attempt+1}]: {e}")

        # 3) MyMemory (free, different provider — often works when Google is blocked)
        try:
            from deep_translator import MyMemoryTranslator
            for src in ("en", "zh-CN", "auto"):
                try:
                    # MyMemory is weaker with multi-line; use single line only
                    kh = MyMemoryTranslator(source=src, target="km").translate(text)
                    if _ok(kh) and kh.strip().lower() != text.strip().lower():
                        return _polish_khmer_line(str(kh))
                except Exception:
                    continue
        except Exception as e:
            last_err = e
            print(f"[translate MyMemory attempt {attempt+1}]: {e}")

        # 4) alternate Google clients5 endpoint
        try:
            import requests
            r = requests.get(
                "https://clients5.google.com/translate_a/t",
                params={
                    "client": "dict-chrome-ex",
                    "sl": "auto",
                    "tl": "km",
                    "q": query,
                },
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=20,
            )
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and data:
                    if isinstance(data[0], list) and data[0]:
                        kh = str(data[0][0]).strip()
                    else:
                        kh = str(data[0]).strip()
                    if _ok(kh):
                        return _polish_khmer_line(_extract_middle(kh))
        except Exception as e:
            last_err = e
            print(f"[translate clients5 attempt {attempt+1}]: {e}")

        if attempt < retries - 1:
            time.sleep(0.6 * (attempt + 1))

    if last_err:
        print(f"[translate gave up after {retries} tries]: {last_err}")
    return ""


def _translate_batch_llm(batch, prev_khmer="", engine="auto", style="recap", prev_source_lines=None):
    """Translate one batch with LLM. Returns list of Khmer strings (same length as batch)."""
    sys_prompt = PROMPT_STYLES.get(style, PROMPT_STYLES["recap"])
    usr_prompt = _build_translate_user_prompt(batch, prev_khmer, style=style, prev_source_lines=prev_source_lines)
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": usr_prompt},
    ]
    raw, model_or_err = _llm_chat_translate(messages, system_prompt=sys_prompt, user_prompt=usr_prompt, engine_choice=engine, style=style)
    if not raw:
        return None, model_or_err

    mapped = _parse_numbered_translations(raw, len(batch))
    result = []
    for i, s in enumerate(batch):
        kh = (mapped[i] if i < len(mapped) else "") or ""
        kh = _polish_khmer_line(kh)
        # If model returned non-Khmer, fix with free backend
        if kh and not _has_khmer(kh) and s.get("text"):
            fixed = _translate_one_google(s["text"])
            if fixed:
                kh = fixed
        if not kh:
            kh = _translate_one_google(s.get("text", "")) or s.get("text", "")
        kh = _apply_glossary_replacement(kh)
        result.append(kh)
    return result, model_or_err


def _translate_to_khmer(segments, progress_state, engine="auto", style="recap", saved=None, save_partial=None):
    """
    Translate each subtitle line to natural spoken Khmer.
    Priority: Gemini AI → OpenAI → Groq LLM → free Google/MyMemory.
    Uses previous-line context so dialogue stays consistent.
    Never hard-fails the whole job.
    saved / save_partial: work finished by an earlier (paused) run, and a callback that stores
    the work done so far — so a resumed job only translates what is still missing.
    """
    if not segments:
        return []
    saved = saved if isinstance(saved, dict) else {}

    cfg = load_config()
    if not engine or engine == "default":
        engine = cfg.get("default_translate_engine", "auto")
    if not style or style == "default":
        style = cfg.get("default_translate_style", "recap")

    has_llm = engine != "google" and bool(
        (cfg.get("GEMINI_API_KEY") or "").strip()
        or (cfg.get("OPENAI_API_KEY") or "").strip()
        or (cfg.get("GROQ_API_KEY") or "").strip()
    )
    out = []

    # --- AI translation (best quality) ---
    # 40 lines per request, several requests in flight. A failed batch is retried with
    # other engines, then free Google — only that batch, never the whole movie.
    if has_llm:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        batch_size = int(cfg.get("translate_batch_size", 60))
        workers = int(cfg.get("translate_workers", 2))
        starts = list(range(0, len(segments), batch_size))
        results = {}
        models_used = set()
        if saved.get("mode") == "llm" and saved.get("batch_size") == batch_size:
            for k, v in (saved.get("batches") or {}).items():
                if int(k) in starts and isinstance(v, list):
                    results[int(k)] = v
            if results:
                models_used.add("saved")

        def _do_batch(bi):
            batch = segments[bi: bi + batch_size]
            ctx = [s["text"] for s in segments[max(0, bi - 3): bi]]
            mapped, model_or_err = _translate_batch_llm(
                batch, "", engine=engine, style=style, prev_source_lines=ctx
            )
            if mapped is None and engine != "auto":
                mapped, model_or_err = _translate_batch_llm(
                    batch, "", engine="auto", style=style, prev_source_lines=ctx
                )
            if mapped is None:
                print(f"[translate batch {bi}] AI failed ({model_or_err}) — free Google for this batch")
                mapped = _translate_lines_google_batch([s.get("text", "") for s in batch])
                model_or_err = "google-free"
            return bi, mapped, model_or_err

        progress_state["status"] = f"Translating to Khmer ({engine} AI · {style} style)..."
        progress_state["percent"] = 55
        done_lines = sum(len(v) for v in results.values())
        if results:
            progress_state["status"] = f"Continuing translation — {done_lines}/{len(segments)} lines saved before..."
        pool = ThreadPoolExecutor(max_workers=workers)
        try:
            futs = [pool.submit(_do_batch, bi) for bi in starts if bi not in results]
            for fut in as_completed(futs):
                bi, mapped, model_used = fut.result()
                results[bi] = mapped
                models_used.add(str(model_used))
                done_lines += len(mapped)
                if save_partial:
                    save_partial({"mode": "llm", "batch_size": batch_size,
                                  "batches": {str(k): v for k, v in results.items()}})
                progress_state["percent"] = 55 + int(30 * done_lines / len(segments))
                progress_state["status"] = (
                    f"Translated {done_lines}/{len(segments)} ({', '.join(sorted(models_used))})"
                )
        finally:
            # on Pause don't wait for batches that haven't started
            pool.shutdown(wait=False, cancel_futures=True)

        for bi in starts:
            batch = segments[bi: bi + batch_size]
            mapped = results.get(bi) or []
            for i, s in enumerate(batch):
                kh = mapped[i] if i < len(mapped) else ""
                out.append({
                    "start": s["start"],
                    "end": s["end"],
                    "text": _apply_glossary_replacement(kh or s["text"]),
                    "source": s["text"],
                })
        khmer_n = sum(1 for s in out if _has_khmer(s["text"]))
        print(f"[translate] AI done — {khmer_n}/{len(out)} Khmer lines ({', '.join(sorted(models_used))})")
        return out

    # --- Free backends (Google / MyMemory / etc.) ---
    # Use previous + next line as light context so dialogue stays more coherent
    import time
    progress_state["status"] = "Translating to Khmer (free Google, with context)..."
    total = len(segments)
    done_before = saved.get("lines") if saved.get("mode") == "free" else None
    for i, s in enumerate(segments):
        if isinstance(done_before, list) and i < len(done_before):
            out.append(done_before[i])
            continue
        if save_partial and i and i % 20 == 0:
            save_partial({"mode": "free", "lines": list(out)})
        progress_state["percent"] = 55 + int(30 * i / max(total, 1))
        progress_state["status"] = f"Translate {i+1}/{total} → ខ្មែរ..."
        prev_txt = segments[i - 1]["text"] if i > 0 else ""
        next_txt = segments[i + 1]["text"] if i + 1 < total else ""
        kh = _translate_one_google(
            s["text"],
            context_prev=prev_txt,
            context_next=next_txt,
        )
        if not kh:
            kh = s["text"]  # last resort — never crash the job
        kh = _apply_glossary_replacement(kh)
        out.append({
            "start": s["start"],
            "end": s["end"],
            "text": kh,
            "source": s["text"],
        })
        if i < total - 1 and total > 5:
            time.sleep(0.25)

    khmer_n = sum(1 for s in out if _has_khmer(s["text"]))
    if khmer_n == 0:
        print(
            "[translate] WARNING: No Khmer script detected. "
            "Tip: set GEMINI_API_KEY or OPENAI_API_KEY in Settings for better AI translation."
        )
        progress_state["status"] = (
            "Translation weak (no Khmer detected) — using original text. Job continues…"
        )
    else:
        print(f"[translate] Free backends OK — {khmer_n}/{len(out)} Khmer lines")
    for item in out:
        item["text"] = _apply_glossary_replacement(item.get("text", ""))
    return out
