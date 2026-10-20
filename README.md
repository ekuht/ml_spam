# Классификация спам-писем (NLP + ML)

классификация email-сообщений как `ham` или `spam` с помощью TF-IDF и калиброванного `LinearSVC`.

## Результаты

- Источники обучения: LingSpam (2 873 письма), SpamAssassin (3 945 писем). Всего 6 818 после подготовки.
- Holdout: 1 364 письма, Accuracy **98,09 %**, Precision класса spam **98,24 %**; FP = 6.
- Внешний тест (отдельный архив): 1 596 писем, Accuracy **93,55 %**, Precision класса spam **99,23 %**, Recall spam **90,36 %**; FP = 7, FN = 96.
- Порог отнесения письма к спаму: **0.60**.
- Подробные метрики: [`metrics.json`](metrics.json).

## Сравнение моделей (5-fold cross-validation)

| Модель | Precision spam | Recall spam | F1 spam |
| --- | ---: | ---: | ---: |
| Logistic Regression | 97,07 % | 95,56 % | 96,31 % |
| Linear SVM | 98,28 % | 96,48 % | 97,37 % |
| Naive Bayes | 99,90 % | 69,15 % | 81,72 % |

Сохранённый ноутбук содержит код, численные результаты и графики; предварительные просмотры содержимого писем из его выводов исключены. Это не меняет код и обучение.

## Архитектура

```
Браузер → Apache :80 → Docker / FastAPI :8000 → TF-IDF + Calibrated LinearSVC
```

Этапы обучения находятся в [`training/Kuhta_Konstantin_Results.ipynb`](training/Kuhta_Konstantin_Results.ipynb). Ноутбук использует архивы `lingspam.zip`, `spamassassin.zip`, `test_task_emails.tar.xz`, предварительно размещённые в рабочем каталоге Google Colab.

## Быстрый запуск (Docker)

```bash
docker build -t spam-classifier .
docker run --rm -p 127.0.0.1:8000:8000 spam-classifier
```

Проверка: http://127.0.0.1:8000/ — форма ввода письма; http://127.0.0.1:8000/docs — Swagger.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"Congratulations, you won a prize!"}'
```

Ответ: `label` (`ham` / `spam`), `spam_probability`, `threshold`.

## VPS: Apache + Docker


Адрес демо: **http://5.104.72.214/**.

### API

| Метод | Путь | Назначение |
| --- | --- | --- |
| `GET` | `/` | Веб-форма |
| `GET` | `/health` | Статус |
| `GET` | `/docs` | Swagger |
| `POST` | `/predict` | Прогноз по `{"text":"..."}` |

## Ограничения

Внешний тест показал 96 пропущенных спам-писем. Модель обучалась преимущественно на англоязычных корпусах; работа на новых языках и с новыми рассылками требует отдельной проверки.
