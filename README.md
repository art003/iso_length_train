# iso_length_train — длина трассы по PDF-изометриям

Офлайн-конвейер: PDF → числа с листа (PyMuPDF) → CatBoost GPU → геометрические гарды → сумма только `include`.
Внешнего API (Groq/Gemini/OpenAI) нет. Cursor не нужен.

## Один раз: окружение

Нужны Python 3.11+, NVIDIA GPU с CUDA (обучение — RTX 3070). Расчёт без GPU тоже идёт: модель уже в `models\iso_clf.pkl`.

```bat
cd путь\к\iso_length_train
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Как считать длины (вкладка Расчёт)

```bat
python app.py
```

Откроется http://127.0.0.1:8765 → **Расчёт** → загрузить PDF → Рассчитать.

Или двойной клик `конвейер_обучения.bat` → пункт «открыть приложение».

Тестовый комплект 10 листов: `data\02_Изометрии_10_листов.pdf`.
Большой исходник `data\Изометрии.pdf` в git не кладётся (тяжёлый).

## Как учить заново (не обязательно)

Holdout **листы 278–307** исходного PDF (= тест_1/2/3) **никогда не входят в train**. `prepare` их пропускает.

```bat
python pipeline.py prepare "C:\путь\Изометрии.pdf"
python pipeline.py annotate
python pipeline.py audit
python pipeline.py train
```

Обучение пишет `models\iso_clf.pkl` и `models\training_report.html`.
В проде классификатор берёт **argmax**, без порога «глянуть». Порог 0.9 на валидации ломал суммы.

Дообучения в проде нет: один раз обучил — считаешь.

## Где что лежит

| Путь | Зачем |
|---|---|
| `models\iso_clf.pkl` | обученная модель, без неё Расчёт не размечает |
| `models\training_report.json` | факты обучения (F1, depth, iteration) |
| `output\Изометрии_без_API\` | разметка 1–277 (нужна, чтобы переучить). В git не входит |
| `output\проверка_мастеру\` | HTML-пакет для проверки длин |
| `docs\Обучение_локальной_модели.pdf` | объяснение для защиты |
| `РАЗМЕТКА.md` | правила классов include / exclude_nested / not_length |

Картинки листа:

- `overlays\*_ids.png` — жёлтые D1, D2… на исходнике;
- `overlays\*_hits.json` — координаты, руками не править;
- `markup\*.png` — цвет: зелёный в сумме, фиолетовый вложенный, серый не длина.

## Перед zip / чужим git

Не коммитить и не класть в архив:

- `.venv\`, `.cursor\`, `.env`, `web\uploads\`
- `data\Изометрии.pdf` (большой исходник)
- `output\Изометрии_без_API\` (если архив для запуска без переобучения — модель достаточно)
- OneDrive `desktop.ini`

Оставить: код, `requirements.txt`, `models\iso_clf.pkl`, `data\02_Изометрии_10_листов.pdf`, `docs\`, `output\проверка_мастеру\`.

Если `iso_clf.pkl` больше ~50 МБ (сейчас ≈0.5 МБ — LFS не нужен):

```bat
git lfs install
git lfs track "models/*.pkl"
```

Это обычный git-репозиторий, не Cursor-hosted и не origin.cursor.com.

## Проверки

```bat
python -m unittest test_guards -v
```
