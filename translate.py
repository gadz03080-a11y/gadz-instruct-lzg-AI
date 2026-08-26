import argparse
import threading
import tempfile
import wave
from pathlib import Path
from typing import Any
import tkinter as tk
from tkinter import ttk
import winsound

import torch
import soundfile as sf
from piper import PiperVoice
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
from transformers import VitsModel

MODEL_DIR = Path(__file__).resolve().parent / "lezgi-nllb-600m-200k-syntetics"
QWEN_MODEL_DIR = Path(__file__).resolve().parent / "models" / "gadz-instruct-lzg-4bit"
QWEN_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
QWEN_DISPLAY_NAME = "gadz-instruct-lzg"
LEZGI_TTS_DIR = Path(__file__).resolve().parent / "models" / "vits-lez-tts"
MODEL_CACHE = {}
QWEN_CACHE = None
TTS_CACHE = {}


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
    if QWEN_CACHE is not None:
        return QWEN_CACHE

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
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
        model.to("cpu")
    model.eval()
    QWEN_CACHE = (tokenizer, model)
    return QWEN_CACHE


def ask_qwen(
    text: str,
    model_dir: str | Path = QWEN_MODEL_DIR,
    history: list[dict[str, str]] | None = None,
    max_new_tokens: int = 128,
) -> str:
    tokenizer, model = load_qwen(model_dir)
    messages = [{"role": "system", "content": "Отвечай по-русски, кратко и понятно. Не переводи ответ на лезгинский."}]
    messages.extend((history or [])[-8:])
    messages.append({"role": "user", "content": text})
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
    return tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()


def stream_qwen(
    text: str,
    model_dir: str | Path = QWEN_MODEL_DIR,
    history: list[dict[str, str]] | None = None,
    on_token=None,
    max_new_tokens: int = 128,
) -> str:
    tokenizer, model = load_qwen(model_dir)
    messages = [{"role": "system", "content": "Отвечай по-русски, кратко и понятно. Не переводи ответ на лезгинский."}]
    messages.extend((history or [])[-8:])
    messages.append({"role": "user", "content": text})
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
    for token in streamer:
        parts.append(token)
        if on_token:
            on_token(token)
    generation_thread.join()
    return "".join(parts).strip()


def answer_in_lezgi(
    text: str,
    translator_model_dir: str | Path = MODEL_DIR,
    qwen_model_dir: str | Path = QWEN_MODEL_DIR,
    history: list[dict[str, str]] | None = None,
    answer_lang: str = "lez_Cyrl",
    on_token=None,
) -> tuple[str, str, str]:
    russian_question = translate(text, translator_model_dir, "lez_Cyrl", "rus_Cyrl")
    russian_answer = stream_qwen(russian_question, qwen_model_dir, history, on_token=on_token)
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


class TranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("gadz-instruct-lzg | Лезгинский AI-чат")
        self.root.geometry("900x680")
        self.root.minsize(620, 480)
        self.root.configure(bg="#eef2f6")
        style = ttk.Style()
        style.configure("App.TFrame", background="#eef2f6")
        style.configure("Header.TFrame", background="#17324d")
        style.configure("Title.TLabel", background="#17324d", foreground="white", font=("Segoe UI", 16, "bold"))
        style.configure("Subtitle.TLabel", background="#17324d", foreground="#c7d5e2")
        style.configure("Status.TLabel", background="#eef2f6", foreground="#5b6573")

        self.model_dir = MODEL_DIR
        self.qwen_model_dir = QWEN_MODEL_DIR
        header = ttk.Frame(root, padding=(18, 14, 18, 14), style="Header.TFrame")
        header.pack(fill="x")
        ttk.Label(header, text="gadz-instruct-lzg", style="Title.TLabel").pack(side="left")
        ttk.Label(header, text="  Лезгинский AI-чат", style="Subtitle.TLabel").pack(side="left", pady=(4, 0))
        ttk.Label(header, text="Язык ответа:").pack(side="right", padx=(10, 6))
        self.answer_combo = ttk.Combobox(header, width=12, state="readonly", values=["Лезгинский", "Русский"])
        self.answer_combo.set("Лезгинский")
        self.answer_combo.pack(side="right")
        self.speak_btn = ttk.Button(header, text="Озвучить ответ", command=self.speak_last_answer, state="disabled")
        self.speak_btn.pack(side="right", padx=(8, 0))
        ttk.Button(header, text="Новый чат", command=self.clear_chat).pack(side="right")

        chat_frame = ttk.Frame(root, padding=(18, 14, 18, 10), style="App.TFrame")
        chat_frame.pack(fill="both", expand=True)
        self.chat_text = tk.Text(
            chat_frame,
            wrap="word",
            state="disabled",
            font=("Segoe UI", 12),
            padx=14,
            pady=12,
            bg="#ffffff",
            relief="solid",
            borderwidth=1,
        )
        scrollbar = ttk.Scrollbar(chat_frame, command=self.chat_text.yview)
        self.chat_text.configure(yscrollcommand=scrollbar.set)
        self.chat_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.chat_text.tag_configure("user", foreground="#174a7e", spacing3=8)
        self.chat_text.tag_configure("bot", foreground="#17613a", spacing3=8)
        self.chat_text.tag_configure("error", foreground="#a32626", spacing3=8)

        input_frame = ttk.Frame(root, padding=(18, 0, 18, 8), style="App.TFrame")
        input_frame.pack(fill="x")
        self.input_text = tk.Text(input_frame, height=3, wrap="word", font=("Segoe UI", 12))
        self.input_text.pack(side="left", fill="both", expand=True)
        self.input_text.bind("<Return>", self.on_enter)
        self.send_btn = ttk.Button(input_frame, text="Отправить", command=self.answer_now)
        self.send_btn.pack(side="right", fill="y", padx=(8, 0))

        self.status_var = tk.StringVar(value="Готово")
        ttk.Label(root, textvariable=self.status_var, style="Status.TLabel").pack(anchor="w", padx=18, pady=(0, 12))

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
        answer_lang = "rus_Cyrl" if self.answer_combo.get() == "Русский" else "lez_Cyrl"
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
                )
                self.russian_history.extend([
                    {"role": "user", "content": russian_question},
                    {"role": "assistant", "content": russian_answer},
                ])
            except Exception as exc:
                result = f"Ошибка: {exc}"
            self.root.after(0, lambda: self.show_result(result, streamed=answer_lang == "rus_Cyrl"))

        threading.Thread(target=worker, daemon=True).start()

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

    def show_result(self, result, streamed=False):
        tag = "error" if result.startswith("Ошибка:") else "bot"
        if streamed:
            self.chat_text.configure(state="normal")
            self.stream_closed = True
            if self.streaming_start is not None:
                self.chat_text.delete(self.streaming_start, "end")
                self.chat_text.insert("end", f"{result}\n\n")
            else:
                self.chat_text.insert("end", f"{result}\n\n")
            self.chat_text.configure(state="disabled")
        elif tag == "bot":
            self.add_message("Ассистент", "", tag)
            self.streaming_answer = ""
            self.stream_closed = False
            self.stream_final_answer(result)
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

    def stream_final_answer(self, text, position=0):
        if position >= len(text):
            self.append_stream("\n\n")
            self.stream_closed = True
            self.status_var.set("Готово")
            self.send_btn.state(["!disabled"])
            self._busy = False
            return
        chunk = text[position:position + 4]
        self.append_stream(chunk)
        self.root.after(22, lambda: self.stream_final_answer(text, position + len(chunk)))

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
    args = parser.parse_args()

    if args.text is not None:
        print(translate(args.text, model_dir=args.model_dir, src_lang=args.src, tgt_lang=args.tgt, max_length=args.max_length))
        return

    root = tk.Tk()
    app = TranslatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
