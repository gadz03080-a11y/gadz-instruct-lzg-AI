import argparse
from html import unescape
import json
import re
import threading
import tempfile
import wave
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import tkinter as tk
from tkinter import ttk
import winsound

import torch
import soundfile as sf
from piper import PiperVoice
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
from transformers import VitsModel

MODEL_DIR = Path(__file__).resolve().parent / "lezgi-nllb-600m-200k-syntetics"
QWEN_MODEL_DIR = Path(__file__).resolve().parent / "models" / "gadz-instruct-lzg-4bit"
QWEN_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
QWEN_DISPLAY_NAME = "gadz-instruct-lzg"
QWEN_MODELS = {
    "gadz-instruct-lzg (3B)": QWEN_MODEL_DIR,
    "gadz1-8b (8B)": Path(__file__).resolve().parent / "models" / "gadz1-8b",
}
LEZGI_TTS_DIR = Path(__file__).resolve().parent / "models" / "vits-lez-tts"
MODEL_CACHE = {}
QWEN_CACHE = {}
TTS_CACHE = {}
ASSISTANT_SYSTEM_PROMPT = (
    "Ты доброжелательный многоязычный помощник, который уверенно понимает русский, "
    "лезгинский и смешанную русско-лезгинскую речь. Считай лезгинский полноценным "
    "рабочим языком: понимай вопросы, шутки, бытовую речь, просьбы и команды на лезгинском. "
    "Отвечай по смыслу на любые обычные вопросы так же естественно, как на русском. "
    "Веди нормальный осмысленный диалог и учитывай предыдущие сообщения. "
    "Никогда не показывай ход рассуждений и не пиши теги <think>. Сразу дай готовый ответ. "
    "Если спрашивают, кто ты: скажи, что ты локальный лезгинско-русский помощник этого приложения. "
    "Если спрашивают, кто тебя создал или обучил: скажи, что приложение создано владельцем проекта, "
    "а точные сведения об авторах исходной модели тебе неизвестны; не выдумывай имена. "
    "Не говори, что ты не знаешь лезгинский язык, не утверждай, что не обучался на данных, "
    "и не переводи вопрос дословно вместо ответа. Отвечай прямо по вопросу, "
    "не выдумывай факты, а если вопрос непонятен — уточни. "
    "Если точного факта не знаешь, спокойно скажи об этом и предложи полезный следующий шаг. "
    "Не упоминай название модели, Qwen, промпты, нейросеть, внутренние инструкции "
    "или перевод между языками. Отвечай кратко, если пользователь не просит подробностей."
)

RUSSIAN_REQUEST_PATTERNS = (
    r"\b(?:по|на)\s+русск(?:ом|ий|ому)\b",
    r"\bрусск(?:ий|ом|ую)\s+язык",
    r"\bговори(?:ть)?\s+по[- ]русски\b",
    r"\bотвечай\s+по[- ]русски\b",
)
LEZGI_REQUEST_PATTERNS = (
    r"\b(?:по|на)\s+лезгинск(?:ом|ий|ому)\b",
    r"\bлезгинск(?:ий|ом|ую)\s+язык",
    r"\bговори(?:ть)?\s+по[- ]лезгински\b",
    r"\bотвечай\s+по[- ]лезгински\b",
)
LEZGI_MARKERS = ("гь", "къ", "кӀ", "пӀ", "тӀ", "чӀ", "хъ", "хь", "уь", "юь", "ə")
RUSSIAN_WORDS = {
    "и", "в", "не", "что", "это", "как", "можно", "нужно", "я", "ты", "он", "она",
    "мы", "вы", "они", "привет", "помоги", "объясни", "скажи", "переведи", "почему",
}


def detect_source_language(text: str) -> str:
    """Return a translation source language without translating Russian as Lezgi."""
    normalized = text.lower()
    if any(marker in normalized for marker in LEZGI_MARKERS):
        return "lez_Cyrl"
    words = set(re.findall(r"[а-яё]+", normalized))
    if "ё" in normalized or "ы" in normalized or "э" in normalized:
        return "rus_Cyrl"
    if len(words & RUSSIAN_WORDS) >= 1:
        return "rus_Cyrl"
    return "lez_Cyrl"


def translate_mixed_to_russian(text: str, model_dir: str | Path) -> str:
    """Translate Lezgi chunks while preserving Russian words in mixed messages."""
    if detect_source_language(text) == "rus_Cyrl":
        return text

    tokens = re.split(r"(\s+)", text)
    if not any(any(marker in token.lower() for marker in LEZGI_MARKERS) for token in tokens):
        return translate(text, model_dir, "lez_Cyrl", "rus_Cyrl")

    translated_parts = []
    lezgi_chunk = []
    for token in tokens:
        if token.isspace():
            if lezgi_chunk:
                lezgi_chunk.append(token)
            else:
                translated_parts.append(token)
            continue
        is_lezgi = any(marker in token.lower() for marker in LEZGI_MARKERS)
        if is_lezgi:
            lezgi_chunk.append(token)
            continue
        if lezgi_chunk:
            translated_parts.append(translate("".join(lezgi_chunk), model_dir, "lez_Cyrl", "rus_Cyrl"))
            lezgi_chunk.clear()
        translated_parts.append(token)
    if lezgi_chunk:
        translated_parts.append(translate("".join(lezgi_chunk), model_dir, "lez_Cyrl", "rus_Cyrl"))
    return "".join(translated_parts)


def requested_answer_language(text: str) -> str | None:
    normalized = text.lower().replace("ё", "е")
    if any(re.search(pattern, normalized) for pattern in RUSSIAN_REQUEST_PATTERNS):
        return "rus_Cyrl"
    if any(re.search(pattern, normalized) for pattern in LEZGI_REQUEST_PATTERNS):
        return "lez_Cyrl"
    return None


def clean_model_output(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.replace("</think>", "").strip()


class SearchResultParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_result = False
        self.in_title = False
        self.in_snippet = False
        self.title = ""
        self.snippet = ""
        self.url = ""
        self.results = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        classes = (attributes.get("class") or "").split()
        if "result" in classes:
            self.in_result = True
        if self.in_result and "result__a" in classes:
            self.in_title = True
            self.url = attributes.get("href") or ""
        if self.in_result and "result__snippet" in classes:
            self.in_snippet = True

    def handle_endtag(self, tag):
        if self.in_title and tag == "a":
            self.in_title = False
        if self.in_snippet and tag in ("a", "div"):
            self.in_snippet = False
        if self.in_result and tag == "div" and self.title and self.snippet:
            self.results.append((self.title.strip(), self.url, self.snippet.strip()))
            self.in_result = False
            self.title = ""
            self.snippet = ""
            self.url = ""

    def handle_data(self, data):
        if self.in_title:
            self.title += data
        elif self.in_snippet:
            self.snippet += data


def search_web(query: str, limit: int = 4) -> str:
    request = Request(
        f"https://html.duckduckgo.com/html/?q={quote_plus(query)}",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    try:
        with urlopen(request, timeout=8) as response:
            parser = SearchResultParser()
            parser.feed(response.read().decode("utf-8", errors="ignore"))
    except Exception:
        parser = SearchResultParser()
    results = parser.results[:limit]
    if not results:
        results = search_bing(query, limit)
    if not results:
        return "Поиск не дал результатов."
    return "\n".join(
        f"[{index}] {title}\nURL: {url}\nОписание: {snippet}"
        for index, (title, url, snippet) in enumerate(results, 1)
    )


def search_bing(query: str, limit: int = 4):
    request = Request(
        f"https://www.bing.com/search?q={quote_plus(query)}",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    try:
        with urlopen(request, timeout=8) as response:
            page = response.read().decode("utf-8", errors="ignore")
    except Exception:
        return []
    results = []
    for block in re.findall(r'<li class="b_algo".*?</li>', page, flags=re.DOTALL):
        link = re.search(r'<h2>\s*<a href="([^"]+)"[^>]*>(.*?)</a>', block, flags=re.DOTALL)
        snippet = re.search(r'<p>(.*?)</p>', block, flags=re.DOTALL)
        if not link:
            continue
        clean = lambda value: re.sub(r"<[^>]+>", "", unescape(value)).strip()
        results.append((clean(link.group(2)), link.group(1), clean(snippet.group(1)) if snippet else ""))
        if len(results) >= limit:
            break
    return results


def extract_source_urls(web_context: str) -> str:
    urls = re.findall(r"^URL:\s*(https?://\S+)", web_context, flags=re.MULTILINE)
    return "\n".join(f"[{index}] {url}" for index, url in enumerate(urls, 1))


def load_model(model_dir: str | Path):
    model_dir = str(model_dir)
    if model_dir not in MODEL_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        if torch.cuda.is_available():
            model.to("cuda")
        model.eval()
        MODEL_CACHE[model_dir] = (tokenizer, model)
    return MODEL_CACHE[model_dir]


def translate(text: str, model_dir: str | Path = MODEL_DIR, src_lang: str = "rus_Cyrl", tgt_lang: str = "lez_Cyrl", max_length: int = 128) -> str:
    if not text or not text.strip():
        return ""

    tokenizer, model = load_model(model_dir)

    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    tokenizer.set_src_lang_special_tokens(src_lang)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=96)
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
            max_length=min(max_length, 96),
            num_beams=1,
        )
    result = tokenizer.decode(generated[0], skip_special_tokens=True)
    return result.strip()


def load_qwen(model_dir: str | Path = QWEN_MODEL_DIR):
    global QWEN_CACHE
    model_key = str(Path(model_dir).resolve())
    if model_key in QWEN_CACHE:
        return QWEN_CACHE[model_key]

    local_model = Path(model_dir)
    has_local_weights = (
        (local_model / "model.safetensors").exists()
        or (local_model / "model.safetensors.index.json").exists()
    )
    model_path = str(local_model) if has_local_weights else QWEN_MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model: Any = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )
    else:
        config = AutoConfig.from_pretrained(model_path)
        config.quantization_config = None
        cpu_dtype = (
            torch.bfloat16
            if torch.backends.cpu.get_cpu_capability() == "AVX512"
            else torch.float32
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=cpu_dtype,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        model.to("cpu")
    model.eval()
    QWEN_CACHE[model_key] = (tokenizer, model)
    return QWEN_CACHE[model_key]


def ask_qwen(
    text: str,
    model_dir: str | Path = QWEN_MODEL_DIR,
    history: list[dict[str, str]] | None = None,
    max_new_tokens: int = 512,
) -> str:
    tokenizer, model = load_qwen(model_dir)
    messages = [{"role": "system", "content": ASSISTANT_SYSTEM_PROMPT}]
    messages.extend(normalize_chat_history(history or [])[-6:])
    messages.append({"role": "user", "content": text})
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    prompt = prompt.replace("<think>\n\n</think>\n\n", "")
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
    answer_tokens = generated[0][inputs["input_ids"].shape[1]:]
    return clean_model_output(tokenizer.decode(answer_tokens, skip_special_tokens=True))


def stream_qwen(
    text: str,
    model_dir: str | Path = QWEN_MODEL_DIR,
    history: list[dict[str, str]] | None = None,
    on_token=None,
    max_new_tokens: int = 512,
) -> str:
    tokenizer, model = load_qwen(model_dir)
    messages = [{"role": "system", "content": ASSISTANT_SYSTEM_PROMPT}]
    messages.extend(normalize_chat_history(history or [])[-6:])
    messages.append({"role": "user", "content": text})
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    prompt = prompt.replace("<think>\n\n</think>\n\n", "")
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_args = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    generation_thread = threading.Thread(target=model.generate, kwargs=generation_args, daemon=True)
    generation_thread.start()
    parts = []
    visible_text = ""
    for token in streamer:
        parts.append(token)
        cleaned_text = clean_model_output("".join(parts))
        if on_token:
            new_text = cleaned_text[len(visible_text):]
            if new_text:
                on_token(new_text)
            visible_text = cleaned_text
    generation_thread.join()
    return clean_model_output("".join(parts))


def answer_in_lezgi(
    text: str,
    translator_model_dir: str | Path = MODEL_DIR,
    qwen_model_dir: str | Path = QWEN_MODEL_DIR,
    history: list[dict[str, str]] | None = None,
    answer_lang: str = "lez_Cyrl",
    on_token=None,
    web_context: str = "",
) -> tuple[str, str, str]:
    source_lang = detect_source_language(text)
    russian_question = text if source_lang == "rus_Cyrl" else translate_mixed_to_russian(text, translator_model_dir)
    model_question = russian_question
    if web_context:
        model_question += (
            "\n\nАктуальная информация из интернета. Используй ее только для ответа "
            "на вопрос и не упоминай технические детали поиска:\n" + web_context
        )
    russian_answer = stream_qwen(model_question, qwen_model_dir, history, on_token=on_token)
    answer_lang = requested_answer_language(text) or answer_lang
    answer = (
        russian_answer
        if answer_lang == "rus_Cyrl"
        else translate(russian_answer, translator_model_dir, "rus_Cyrl", answer_lang)
    )
    return russian_question, russian_answer, answer


def load_tts(language: str):
    if language in TTS_CACHE:
        return TTS_CACHE[language]

    if language == "lez_Cyrl":
        tokenizer = AutoTokenizer.from_pretrained(str(LEZGI_TTS_DIR), local_files_only=True)
        model = VitsModel.from_pretrained(str(LEZGI_TTS_DIR), local_files_only=True)
        model.eval()
        tts = ("vits", tokenizer, model, 16000)
    else:
        voice = PiperVoice.load(
            str(Path(__file__).resolve().parent / "models" / "piper-ru" / "ru_RU-denis-medium.onnx")
        )
        tts = ("piper", voice, None, 22050)
    TTS_CACHE[language] = tts
    return tts


def speak_text(text: str, language: str):
    if not text.strip():
        return
    tts: Any = load_tts(language)
    temp_path = Path(tempfile.gettempdir()) / f"gadz_tts_{language}.wav"
    if tts[0] == "piper":
        with wave.open(str(temp_path), "wb") as wav_file:
            tts[1].synthesize_wav(text, wav_file)
        winsound.PlaySound(str(temp_path), winsound.SND_FILENAME)
        return
    if tts[0] == "vits":
        _, tokenizer, model, sample_rate = tts
        inputs = tokenizer(text, return_tensors="pt")
        with torch.inference_mode():
            audio = model(**inputs).waveform[0].cpu().numpy()
        sf.write(temp_path, audio, sample_rate)
        winsound.PlaySound(str(temp_path), winsound.SND_FILENAME)


class ChatApiHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict[str, Any]):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.end_headers()

    def do_POST(self):
        if self.path not in ("/api/chat", "/api/speak"):
            self._send_json(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length).decode("utf-8"))
            text = str(request.get("text", "")).strip()
            if not text:
                self._send_json(400, {"error": "Введите сообщение"})
                return
            if self.path == "/api/speak":
                language = request.get("language", "rus")
                speak_text(text, language)
                self._send_json(200, {"ok": True})
                return
            answer_lang = request.get("answer_lang", "rus_Cyrl")
            model_dir = QWEN_MODELS.get(request.get("model", ""), QWEN_MODEL_DIR)
            history = normalize_chat_history(request.get("history", []))[-6:]
            web_context = search_web(text) if request.get("online") else ""
            _, _, answer = answer_in_lezgi(text, MODEL_DIR, model_dir, history, answer_lang, web_context=web_context)
            self._send_json(200, {"answer": answer, "sources": extract_source_urls(web_context)})
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def log_message(self, format, *args):
        return


def run_server(host: str = "127.0.0.1", port: int = 8765):
    server = ThreadingHTTPServer((host, port), ChatApiHandler)
    print(f"Chat API: http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Chat API stopped")
    finally:
        server.server_close()


def normalize_chat_history(history: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized = []
    for item in history:
        role = item.get("role")
        content = item.get("content", item.get("text", ""))
        if role in ("user", "assistant") and content:
            normalized.append({"role": role, "content": str(content)})
    return normalized


class TranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("gadz-instruct-lzg")
        self.root.geometry("1180x760")
        self.root.minsize(760, 540)
        self.root.configure(bg="#212121")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("App.TFrame", background="#212121")
        style.configure("Sidebar.TFrame", background="#171717")
        style.configure("Header.TFrame", background="#212121")
        style.configure("Title.TLabel", background="#171717", foreground="#f4f4f4", font=("Segoe UI", 15, "bold"))
        style.configure("Subtitle.TLabel", background="#171717", foreground="#9b9b9b", font=("Segoe UI", 9))
        style.configure("Status.TLabel", background="#212121", foreground="#8e8e8e", font=("Segoe UI", 9))
        style.configure("Meta.TLabel", background="#212121", foreground="#a0a0a0", font=("Segoe UI", 9))
        style.configure("Sidebar.TButton", background="#171717", foreground="#d6d6d6", borderwidth=0, padding=(12, 9), anchor="w")
        style.map("Sidebar.TButton", background=[("active", "#2a2a2a")])
        style.configure("Modern.TButton", background="#2f2f2f", foreground="#f2f2f2", borderwidth=0, padding=(10, 7))
        style.map("Modern.TButton", background=[("active", "#3f3f3f")])
        style.configure("Modern.TCombobox", fieldbackground="#2f2f2f", background="#2f2f2f", foreground="#f2f2f2", borderwidth=0)
        style.configure("Modern.TCheckbutton", background="#212121", foreground="#c7c7c7")

        self.model_dir = MODEL_DIR
        self.qwen_model_dir = QWEN_MODEL_DIR
        self.internet_var = tk.BooleanVar(value=False)
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=1)

        sidebar = ttk.Frame(root, width=220, style="Sidebar.TFrame", padding=(14, 18))
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)
        ttk.Label(sidebar, text="gadz", style="Title.TLabel").pack(anchor="w")
        ttk.Label(sidebar, text="ЛЕЗГИНСКИЙ AI", style="Subtitle.TLabel").pack(anchor="w", pady=(0, 24))
        ttk.Button(sidebar, text="＋  Новый чат", style="Sidebar.TButton", command=self.clear_chat).pack(fill="x", pady=(0, 12))
        ttk.Label(sidebar, text="ЧАТЫ", style="Subtitle.TLabel").pack(anchor="w", padx=12, pady=(12, 8))
        ttk.Label(sidebar, text="Текущий диалог", style="Sidebar.TLabel").pack(fill="x", padx=12, pady=5)
        ttk.Label(sidebar, text="\nЛокальная модель\nБез передачи текста", style="Subtitle.TLabel").pack(side="bottom", anchor="w", padx=12)

        content = ttk.Frame(root, style="App.TFrame")
        content.grid(row=0, column=1, sticky="nsew")
        content.grid_rowconfigure(1, weight=1)
        content.grid_columnconfigure(0, weight=1)

        header = ttk.Frame(content, padding=(28, 18, 28, 14), style="Header.TFrame")
        header.grid(row=0, column=0, sticky="ew")
        ttk.Label(header, text="Новый разговор", foreground="#f4f4f4", background="#212121", font=("Segoe UI", 16, "bold")).pack(side="left")
        ttk.Label(header, text="  Лезгинский AI-чат", style="Meta.TLabel").pack(side="left", pady=(4, 0))
        self.speak_btn = ttk.Button(header, text="Озвучить", style="Modern.TButton", command=self.speak_last_answer, state="disabled")
        self.speak_btn.pack(side="right", padx=(12, 0))
        ttk.Label(header, text="Язык:", style="Meta.TLabel").pack(side="right", padx=(12, 6))
        self.answer_combo = ttk.Combobox(header, width=12, state="readonly", values=["Лезгинский", "Русский"], style="Modern.TCombobox")
        self.answer_combo.set("Лезгинский")
        self.answer_combo.pack(side="right")
        ttk.Label(header, text="Модель:", style="Meta.TLabel").pack(side="right", padx=(16, 6))
        self.model_combo = ttk.Combobox(
            header, width=20, state="readonly", values=list(QWEN_MODELS), style="Modern.TCombobox"
        )
        self.model_combo.set("gadz-instruct-lzg (3B)")
        self.model_combo.pack(side="left")
        self.model_combo.bind("<<ComboboxSelected>>", self.change_model)
        ttk.Checkbutton(header, text="Интернет", variable=self.internet_var, style="Modern.TCheckbutton").pack(side="right", padx=(0, 16))

        chat_frame = ttk.Frame(content, padding=(28, 8, 28, 10), style="App.TFrame")
        chat_frame.grid(row=1, column=0, sticky="nsew")
        chat_frame.grid_rowconfigure(0, weight=1)
        chat_frame.grid_columnconfigure(0, weight=1)
        self.chat_text = tk.Text(
            chat_frame,
            wrap="word",
            state="disabled",
            font=("Segoe UI", 11),
            padx=22,
            pady=18,
            bg="#212121",
            fg="#ececec",
            insertbackground="#ffffff",
            selectbackground="#3b82f6",
            relief="flat",
            borderwidth=0,
        )
        scrollbar = ttk.Scrollbar(chat_frame, command=self.chat_text.yview, orient="vertical")
        self.chat_text.configure(yscrollcommand=scrollbar.set)
        self.chat_text.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.chat_text.tag_configure("user", foreground="#78b7ff", spacing3=10)
        self.chat_text.tag_configure("bot", foreground="#f1f1f1", spacing3=10)
        self.chat_text.tag_configure("source", foreground="#8fa3b8", spacing3=4)
        self.chat_text.tag_configure("error", foreground="#ff7b72", spacing3=8)

        composer = tk.Frame(content, bg="#2f2f2f", highlightthickness=1, highlightbackground="#454545", highlightcolor="#5b5b5b")
        composer.grid(row=2, column=0, sticky="ew", padx=28, pady=(0, 8))
        self.input_text = tk.Text(composer, height=3, wrap="word", font=("Segoe UI", 11), bg="#2f2f2f", fg="#f1f1f1", insertbackground="#ffffff", relief="flat", borderwidth=0, padx=14, pady=12)
        self.input_text.pack(side="left", fill="both", expand=True)
        self.input_text.bind("<Return>", self.on_enter)
        self.send_btn = ttk.Button(composer, text="➤", width=3, style="Modern.TButton", command=self.answer_now)
        self.send_btn.pack(side="right", fill="y", padx=(0, 8), pady=8)

        self.status_var = tk.StringVar(value="Готово")
        ttk.Label(content, textvariable=self.status_var, style="Status.TLabel").grid(row=3, column=0, sticky="w", padx=30, pady=(0, 12))

        self._busy = False
        self.russian_history = []
        self.last_answer = ""
        self.streaming_answer = ""
        self.streaming_start = None
        self.stream_closed = False

    def on_enter(self, event):
        if not event.state & 0x0001:
            self.answer_now()
            return "break"
        return None

    def answer_now(self):
        text = self.input_text.get("1.0", "end").strip()
        if not text or self._busy:
            return

        self._busy = True
        self.qwen_model_dir = QWEN_MODELS[self.model_combo.get()]
        use_internet = self.internet_var.get()
        selected_answer_lang = "rus_Cyrl" if self.answer_combo.get() == "Русский" else "lez_Cyrl"
        answer_lang = requested_answer_language(text) or selected_answer_lang
        self.status_var.set("Перевожу и формирую ответ...")
        self.send_btn.state(["disabled"])
        self.input_text.delete("1.0", "end")
        self.add_message("Вы", text, "user")
        if answer_lang == "rus_Cyrl":
            self.add_message("Ассистент", "", "bot")
            self.streaming_start = self.chat_text.index("end-1c")
            self.stream_closed = False
        self.streaming_answer = ""
        history = list(self.russian_history)
        web_context = search_web(text) if use_internet else ""
        sources = extract_source_urls(web_context)

        def worker():
            try:
                def on_token(token):
                    self.root.after(0, lambda value=token: self.append_stream(value))

                russian_question, russian_answer, result = answer_in_lezgi(
                    text,
                    self.model_dir,
                    self.qwen_model_dir,
                    history,
                    answer_lang,
                    on_token if answer_lang == "rus_Cyrl" else None,
                    web_context,
                )
                self.russian_history.extend([
                    {"role": "user", "content": russian_question},
                    {"role": "assistant", "content": russian_answer},
                ])
            except Exception as exc:
                result = f"Ошибка: {exc}"
            self.root.after(
                0,
                lambda: self.show_result(
                    result, streamed=answer_lang == "rus_Cyrl", sources=sources
                ),
            )

        threading.Thread(target=worker, daemon=True).start()

    def change_model(self, event=None):
        if not self._busy:
            self.qwen_model_dir = QWEN_MODELS[self.model_combo.get()]
            self.status_var.set(f"Выбрана модель: {self.model_combo.get()}")

    def add_message(self, author, text, tag):
        self.chat_text.configure(state="normal")
        self.chat_text.insert("end", f"{author}:\n", tag)
        if text:
            self.chat_text.insert("end", f"{text}\n\n")
        self.chat_text.configure(state="disabled")
        self.chat_text.see("end")

    def append_stream(self, value):
        if self.stream_closed:
            return
        self.streaming_answer += value
        self.chat_text.configure(state="normal")
        self.chat_text.insert("end", value)
        self.chat_text.configure(state="disabled")
        self.chat_text.see("end")

    def show_result(self, result, streamed=False, sources=""):
        tag = "error" if result.startswith("Ошибка:") else "bot"
        if streamed:
            self.chat_text.configure(state="normal")
            self.stream_closed = True
            if self.streaming_start is not None:
                self.chat_text.delete(self.streaming_start, "end")
                self.chat_text.insert("end", f"{result}\n\n")
            else:
                self.chat_text.insert("end", f"{result}\n\n")
            if sources:
                self.chat_text.insert("end", f"Источники:\n{sources}\n\n", "source")
            self.chat_text.configure(state="disabled")
        elif tag == "bot":
            self.add_message("Ассистент", "", tag)
            self.streaming_answer = ""
            self.stream_closed = False
            self.stream_final_answer(result, sources=sources)
        else:
            self.add_message("Ассистент", result, tag)
        self.last_answer = "" if tag == "error" else result
        self.speak_btn.state(["!disabled"] if self.last_answer else ["disabled"])
        if streamed or tag == "error":
            self.status_var.set("Готово")
            self.send_btn.state(["!disabled"])
            self._busy = False
        else:
            self.status_var.set("Вывожу ответ...")

    def stream_final_answer(self, text, position=0, sources=""):
        if position >= len(text):
            self.append_stream("\n\n")
            if sources:
                self.add_message("Источники", sources, "source")
            self.stream_closed = True
            self.status_var.set("Готово")
            self.send_btn.state(["!disabled"])
            self._busy = False
            return
        chunk = text[position:position + 4]
        self.append_stream(chunk)
        self.root.after(
            22, lambda: self.stream_final_answer(text, position + len(chunk), sources)
        )

    def speak_last_answer(self):
        if not self.last_answer or self._busy:
            return
        language = "rus" if self.answer_combo.get() == "Русский" else "lez_Cyrl"
        self._busy = True
        self.speak_btn.state(["disabled"])
        self.status_var.set("Озвучиваю русский ответ..." if language == "rus" else "Озвучиваю лезгинский ответ...")

        def worker():
            try:
                speak_text(self.last_answer, language)
                status = "Готово"
            except Exception as exc:
                status = f"Ошибка озвучки: {exc}"
            self.root.after(0, lambda: self.finish_speaking(status))

        threading.Thread(target=worker, daemon=True).start()

    def finish_speaking(self, status):
        self.status_var.set(status)
        self._busy = False
        self.speak_btn.state(["!disabled"] if self.last_answer else ["disabled"])

    def clear_chat(self):
        if self._busy:
            return
        self.russian_history.clear()
        self.last_answer = ""
        self.speak_btn.state(["disabled"])
        self.chat_text.configure(state="normal")
        self.chat_text.delete("1.0", "end")
        self.chat_text.configure(state="disabled")
        self.status_var.set("Новый чат")


def main():
    parser = argparse.ArgumentParser(description="Local translator with a simple GUI")
    parser.add_argument("text", nargs="?", help="Optional text to translate in CLI mode")
    parser.add_argument("--model-dir", default=str(MODEL_DIR), help="Path to the model folder")
    parser.add_argument("--src", default="rus_Cyrl", help="Source language code")
    parser.add_argument("--tgt", default="lez_Cyrl", help="Target language code")
    parser.add_argument("--max-length", type=int, default=128, help="Max generation length")
    parser.add_argument("--server", action="store_true", help="Run local API for the web interface")
    parser.add_argument("--port", type=int, default=8765, help="Local API port")
    args = parser.parse_args()

    if args.server:
        run_server(port=args.port)
        return

    if args.text is not None:
        print(translate(args.text, model_dir=args.model_dir, src_lang=args.src, tgt_lang=args.tgt, max_length=args.max_length))
        return

    root = tk.Tk()
    app = TranslatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
