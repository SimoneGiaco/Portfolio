install:
	pip install --upgrade pip &&\
		pip install -r RAG/requirements.txt

test:
	python -m pytest --nbval RAG/PDFarXivQA.ipynb

all: install test 