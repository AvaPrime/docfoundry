VENV?=.venv
PY?=$(VENV)/bin/python
PIP?=$(VENV)/bin/pip

setup:
	python -m venv $(VENV) && $(PIP) install -U pip && $(PIP) install -r requirements.txt

serve-docs:
	mkdocs serve -a 127.0.0.1:8000

crawl-html:
	$(PY) pipelines/html_ingest.py sources/openrouter.yaml

index:
	$(PY) indexer/build_index.py

api:
	uvicorn server.rag_api:app --reload --port 8001

all: crawl-html index api
