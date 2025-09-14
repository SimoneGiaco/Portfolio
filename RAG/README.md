We provide two Jupyter notebooks which contain the python code to perform RAG with Langchain. \
The problem we consider is to let a llm answer questions about a document provided by the user. With Langchain libraries we split the text into small chunks, store them in a Chromadb collection. \
When we ask a question Chromadb selects the most relevant chunks according to cosine similarity and these are added to the prompt for the llm. \
We consider the case of pdf files and files from the arXiv website (preprints of scientific papers). We include in both files a Gradio UI to interact with the LLM. \
The LLMs we have chosen are GGUF quantized, so that they can be run on CPU. They are trained for text-generation tasks and can be found on Huggingface.

1) In the 'PDFQA' file we consider PDF files only. The chosen LLM is the mistral "TheBloke--zephyr-7B-beta-GGUF". For each question we select 3 chunks of 350 characters each to fit the context window of 512 tokens.

2) In the 'PDFarXivQA' the Gradio interface can support both pdf files uploads and uploads from the arXiv by providing the arXiv number of the preprint. \
The chosen LLM is the microsoft "Phi-3-mini-4k-instruct-q4.gguf" which is faster to run (3B parameters) and has an editable context window (set in the file to 4096 tokens). For each question we select 5 chunks of size 500 characters. \
To run the code one needs first to download the model (select the q4.gguf version) from Huggingface [here] https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/tree/main
