# InfoQuery AI (Your intelligent document companion)

InfoQuery AI is your intelligent document companion powered by OpenAI LLM and langchain. Upload PDF, DOC, or DOCX files, and let InfoQuery AI extract the text, perform advanced text matching operations, and answer your questions based on the document's content. Experience effortless information retrieval and tap into the vast knowledge within your documents with InfoQuery AI.

The application uses a LLM to generate a response about your Document. The LLM will not answer questions unrelated to the document.

## How it works

The application reads the document and splits the text into smaller chunks that can be then fed into a LLM. It uses OpenAI embeddings to create vector representations of the chunks. The application then finds the chunks that are semantically similar to the question that the user asked and feeds those chunks to the LLM to generate a response.

The application uses Streamlit to create the GUI and Langchain to deal with the LLM.

## Installation

To install the repository, please clone this repository and install the requirements:

You will also need to add your OpenAI API key to the `.env` file.

## Usage

To use the application, run the `app.py` file with the streamlit CLI (after having installed streamlit): 

```
streamlit run app.py
```

## Contributing

This repository is for educational purposes only and is not intended to receive further contributions. I have created this for self learning purpose. I have followed the following Video and tried this

'https://www.youtube.com/watch?v=wUAUdEw5oxM&list=PLMVV8yyL2GN_n41v1ESBvDHwMbYYhlAh1&index=3'


## References
I have also followed these documents

https://faiss.ai/
https://python.langchain.com/en/latest/use_cases/question_answering.html
https://github.com/gkamradt/langchain-tutorials/blob/main/data_generation/Ask%20A%20Book%20Questions.ipynb
https://bennycheung.github.io/ask-a-book-questions-with-langchain-openai