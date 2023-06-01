
# Using python-dotenv to Load Env variables

import codecs
import docx2txt
import PyPDF2
import pandas as pd

import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

def main():
    load_dotenv()

    st.set_page_config(page_title="Ask anything about your Document")
    st.markdown("<h1 style='color: green;'>DocInsight Plus</h1>", unsafe_allow_html=True)
    #st.header("DocInsight Plus")
    st.markdown("<h2 style='color: blue;'>Discover, analyze, and interact with your documents</h2>", unsafe_allow_html=True)
    #st.subheader("Discover, analyze, and interact with your documents")
    

    # upload file
    file = st.file_uploader("Upload your document here [PDF,DOC,DOCX, CSV, TXT]", type=["pdf", "doc", "docx", "TXT", "CSV"])

    # extract the text from the file
    if file is not None:

        file_ext = file.name.split(".")[-1]
        text = ""
        if file_ext == "pdf":
            pdf_reader = PyPDF2.PdfReader(file)

            for page in pdf_reader.pages:
                text += page.extract_text()
                
        elif file_ext == "docx":
            text = docx2txt.process(file)
        
        elif file_ext == "doc":
            # Convert .doc to .docx before extracting text
            docx_file = st._maybe_convert_to_docx(file)

            if docx_file:
                text = docx2txt.process(docx_file)
            else:
                st.error("Failed to convert .doc file to .docx")
        
        elif file_ext == "csv":
            df = pd.read_csv(file)
            text = df.to_string(index=False)
        
        elif file_ext == "txt":
            text = codecs.decode(file.read(), encoding='utf-8')

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