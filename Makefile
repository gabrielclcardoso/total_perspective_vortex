RUN ?= uv run

SUBJECT ?= 1
EXP_RUN ?= 4

.PHONY: setup train predict evaluate visualize notebook clean fclean

setup:
	uv sync

train:
	$(RUN) python mybci.py $(SUBJECT) $(EXP_RUN) train

predict:
	$(RUN) python mybci.py $(SUBJECT) $(EXP_RUN) predict

evaluate:
	$(RUN) python mybci.py

visualize:
	$(RUN) python -m tpv.visualize

notebook:
	$(RUN) --with jupyter jupyter lab notebooks/

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf models/*.joblib plots/*.png

fclean: clean
	rm -rf .venv data/*
