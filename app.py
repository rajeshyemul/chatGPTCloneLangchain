
# Using python-dotenv to Load Env variables

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

def main():
    load_dotenv()

    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")

    # upload file
    pdf = st.file_uploader("Upload your document here", type= "pdf")

    # extract the text from the file
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # split text into chunks
        text_chunks = CharacterTextSplitter (
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_chunks.split_text(text)
    
        # create embedding and from that create the knowledge base to ask the questions
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # input box to ask the questions
        user_question = st.text_input("Ask a question about your document : ")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)

            st.write(response)

if __name__ == '__main__':
    main()