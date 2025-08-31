# ========== ОБУЧЕНИЕ ==========
train:
	python -m src.models.train

# ========== ИНФЕРЕНС В КОНСОЛИ ==========
infer:
	python -c "from src.models.infer import load_router; \
infer = load_router(); \
sample = ['не работает оплата', 'хочу подписку на год', 'чтоооо?']; \
[print(f'[{p.label_id} | {p.label_name}] → {t}') for p, t in zip(infer(sample), sample)]"

# ========== ИНТЕРАКТИВНЫЙ ИНФЕРЕНС ==========
test_infer:
	python -m src.serving.cli

# ========== FASTAPI ==========
api:
	uvicorn src.serving.api_fastapi:app --reload

# ========== УДАЛЕНИЕ АРТЕФАКТОВ И ГРАФИКОВ ==========
clean:
	rm -f models/artifacts/*.joblib
	rm -f models/artifacts/*.json
	rm -f reports/figures/*.png

# ========== ПОМОЩЬ ==========
help:
	@echo "🧠 Команды:"
	@echo "  make train       — обучение модели и сохранение артефактов"
	@echo "  make infer       — быстрый batch-инференс (встроенные примеры)"
	@echo "  make test_infer  — интерактивный инференс (вводите текст строкой)"
	@echo "  make api         — запуск FastAPI сервера на http://127.0.0.1:8000"
	@echo "  make clean       — удалить артефакты и графики"
