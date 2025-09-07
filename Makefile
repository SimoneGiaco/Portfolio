install:
	pip install --upgrade pip &&\
		pip install -r RAG/requirements.txt

test:
	python -m pytest --nbval RAG/PDFarXivQA.ipynb

format:	
	black *.py 

lint:
	pylint --disable=R,C main.py

all: install lint test format 
