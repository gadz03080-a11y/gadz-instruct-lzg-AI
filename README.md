# gadz-instruct-lzg

Локальный чат-переводчик для русского и лезгинского языков. Приложение работает на Windows и не отправляет текст во внешний API.

## Возможности

- перевод сообщений с лезгинского на русский;
- генерация ответа локальной instruct-моделью;
- выбор языка ответа: русский или лезгинский;
- потоковый вывод ответа;
- русская озвучка через Piper;
- лезгинская озвучка через VITS;
- история текущего диалога в памяти приложения.

## Требования

- Windows 10 или новее;
- Python 3.11;
- видеокарта NVIDIA рекомендуется для быстрой работы;
- локальные файлы моделей из раздела ниже.

## Установка

Клонируйте репозиторий:

```powershell
git clone https://github.com/gadz03080-a11y/gadz-instruct-lzg-AI.git
cd gadz-instruct-lzg-AI
```

Создайте виртуальное окружение и установите зависимости:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Запуск:

```powershell
.\run_translator.bat
```

Или напрямую:

```powershell
.venv\Scripts\python.exe translate.py
```

## Модели

### Скачать модели

Все рабочие модели находятся в публичном репозитории Hugging Face:

https://huggingface.co/gadz777/gadz-instruct-lzg-models

Из корня проекта (`C:\путь\к\gadz-instruct-lzg-AI`) скачайте их командой:

```powershell
py -3.11 -m pip install -U huggingface-hub
py -3.11 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='gadz777/gadz-instruct-lzg-models', repo_type='model', local_dir='.', allow_patterns=['models/**', 'lezgi-nllb-600m-200k-syntetics/**'], ignore_patterns=['**/.cache/**', '**/checkpoint/**'])"
```

Команда автоматически создаст нужные каталоги рядом с `translate.py`. Для скачивания требуется около 29 ГБ свободного места.

Если скачиваете файлы вручную, сохраните структуру каталогов без переименований:

| Каталог | Назначение |
| --- | --- |
| `lezgi-nllb-600m-200k-syntetics/` | Перевод между русским и лезгинским |
| `models/gadz-instruct-lzg-4bit/` | Локальная instruct-модель для генерации ответов |
| `models/vits-lez-tts/` | Лезгинская озвучка |
| `models/piper-ru/` | Русская озвучка |

Дополнительная модель `models/gadz1-8b/` нужна только для выбора варианта `gadz1-8b (8B)`. Для обычного запуска достаточно остальных каталогов.

Русская Piper-модель должна содержать:

```text
ru_RU-denis-medium.onnx
ru_RU-denis-medium.onnx.json
```

Каждая модель может иметь собственную лицензию. Проверьте условия распространения весов отдельно.

## Как работает приложение

```text
сообщение на лезгинском
        ↓
NLLB: лезгинский → русский
        ↓
gadz-instruct-lzg: генерация ответа
        ↓
NLLB: русский → лезгинский
```

Если выбран русский язык ответа, последний перевод не выполняется.

## Запуск готового приложения Windows

После установки Python и размещения моделей дважды запустите `run_gadz.vbs`. Скрипт автоматически:

1. перейдёт в папку проекта;
2. запустит локальный сервер `translate.py --server`;
3. откроет `gadz.exe`.

Если виртуальное окружение не создано, скрипт использует доступный в `PATH` Python. Для разработки можно запустить интерфейс напрямую:

```powershell
.venv\Scripts\python.exe translate.py
```

Для запуска веб-интерфейса отдельно:

```powershell
.venv\Scripts\python.exe translate.py --server
```

## Лицензия

Исходный код распространяется по лицензии MIT. Лицензии исходных моделей имеют собственные условия и не заменяются лицензией проекта.
