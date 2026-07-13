import os
import spacy #spaCy is used to process and understand human language
import requests #It allows your Python program to communicate with web servers and APIs.
import tempfile #used to create temporary files.
import pandas as pd
import streamlit as st # helps create user interface
from dotenv import load_dotenv #It is commonly used to load environment variables from a .env file into your Python program.
from collections import Counter # is used to count how many times each item appears in a list or collection.
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings,)
from langchain_chroma import Chroma
from google import genai
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import UnstructuredFileLoader
load_dotenv()
nlp = spacy.load("en_core_web_sm") #Small English language model trained on web text
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
st.set_page_config(page_title="Semantic Search Engine",layout="wide")
st.title("Semantic Search Engine")
url = st.text_input("enter a website URL")
uploaded_files = st.file_uploader(
    "Upload Documents",
    accept_multiple_files=True,
    type=[
        "pdf","doc","docx","txt","md","rtf","log",
        "csv","xls","xlsx","ppt","pptx","json","xml",
        "html","htm","py","java","cpp","c","cs",
        "js","ts","css","php",
        "sql"])
if uploaded_files or url.strip():
    all_docs = []
    for uploaded_file in (uploaded_files or []):

        suffix = os.path.splitext(uploaded_file.name)[1].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            file_path = tmp.name

        if suffix == ".pdf":
            loader = PyPDFLoader(file_path)
        elif suffix in [".txt", ".log"]:
            loader = TextLoader(file_path)
        elif suffix == ".csv":
            loader = CSVLoader(file_path)
        elif suffix == ".docx":
            loader = Docx2txtLoader(file_path)
        elif suffix in [".xlsx", ".xls"]:
            loader = UnstructuredExcelLoader(file_path)
        elif suffix in [".ppt", ".pptx"]:
            loader = UnstructuredPowerPointLoader(file_path)
        elif suffix in [".html", ".htm"]:
            loader = UnstructuredHTMLLoader(file_path)
        elif suffix == ".md":
            loader = UnstructuredMarkdownLoader(file_path)
        elif suffix == ".json":
            loader = JSONLoader(
                file_path=file_path,
                jq_schema=".",
                text_content=False
            )
        elif suffix in [
            ".xml", ".rtf",
            ".py", ".java", ".cpp", ".c", ".cs",
            ".js", ".ts", ".css", ".php", ".sql",
            ]:
            loader = UnstructuredFileLoader(file_path)
        else:
            st.warning(f"Unsupported file: {uploaded_file.name}")
            continue

        st.write("Loading:", uploaded_file.name)
        st.write("Extension:", suffix)

        # Load the document
        docs = loader.load()

        # Replace temporary path with original filename
        for doc in docs:
            doc.metadata["source"] = uploaded_file.name
            doc.metadata["file_name"] = uploaded_file.name

        # Add to all documents
        all_docs.extend(docs)

    # Load website
    if url.strip():
        try:
            web_docs = WebBaseLoader(url.strip()).load()
            for doc in web_docs:
                doc.metadata["source"] = url.strip()
                doc.metadata["file_name"] = url.strip()
            all_docs.extend(web_docs)
            st.success("Website loaded successfully.")
        except Exception as e:
            st.error(f"Unable to load website: {e}")

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    # Split all documents into chunks
    chunks = text_splitter.split_documents(all_docs)
    for chunk in chunks:
        if "file_name" in chunk.metadata:
            chunk.metadata["source"] = chunk.metadata["file_name"]
    
    print("Chunks Created:", len(chunks))
    print(f"Total Documents : {len(all_docs)}")
    if len(chunks) == 0:
            st.warning("Please upload at least one supported document.")
            st.stop()
    counter = Counter()#Creates an empty counter to keep track of how many chunks each file has.
    for chunk in chunks:
      counter[chunk.metadata.get("source", "Unknown")] += 1 # Gets the file name from the chunk’s metadata and Increases that file’s count by 1.
    print("\n Chunks Per File ")
    for file, count in counter.items():
        print(f"{file}: {count}")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = Chroma.from_documents(documents=chunks,embedding=embeddings)
    retriever = vectorstore.as_retriever(search_type="mmr",search_kwargs={"k": 5})
    prompt = ChatPromptTemplate.from_template(""" 
                                                  You are an intelligent AI Research Assistant and Semantic Search Engine.
                                                  Your primary task is to answer the user's question using the retrieved context provided below.
                                                  The retrieved context may come from one or more sources, including:
                                                  - PDF documents
                                                  - Word documents (.doc/.docx)
                                                  - PowerPoint presentations (.ppt/.pptx)
                                                  - Text files (.txt, .md, .log)
                                                  - CSV and Excel files
                                                  - HTML pages
                                                  - Source code files
                                                  - JSON, XML, YAML, configuration files
                                                  - URLs and web pages
                                                  Retrieved Context:
                                                  {context}
                                                  User Question:
                                                  {question}
                                                  Instructions:
                                                  1. Read the retrieved context carefully before answering.
                                                  2. If the answer is fully available in the retrieved context:
                                                  - Answer ONLY using the retrieved information.
                                                  - Do not invent or assume missing facts.
                                                  - Combine information from multiple retrieved documents when necessary.
                                                  3. If the retrieved context is only partially relevant:
                                                  - Use the available context first.
                                                  - Then use your own general knowledge to complete the answer.
                                                  - Clearly state:
                                                  "Note: Parts of this answer are based on my general knowledge because the uploaded documents do not contain complete information."
                                                  4. If the retrieved context does not contain the answer at all:
                                                  - Say that the uploaded documents do not contain enough information.
                                                  - Then provide the best possible answer using your general knowledge.
                                                  - Clearly mention that the answer is not based on the uploaded documents.
                                                  5. If multiple uploaded documents contain different or conflicting information:
                                                  - Mention the conflict.
                                                  - Identify which source says what, if possible.
                                                  - Do not merge conflicting facts into a single statement.
                                                  6. If the question is ambiguous:
                                                  - Ask a brief clarifying question before answering.
                                                  7. Keep the answer:
                                                  - Accurate
                                                  - Well structured
                                                  - Easy to understand
                                                  - Concise unless the user requests a detailed explanation.
                                                  8. Use bullet points, numbered lists, or tables whenever they improve readability.
                                                  9. At the end of every response, provide:
                                                  Related Topics:
                                                  - Topic 1
                                                  - Topic 2
                                                  - Topic 3
                                                  - Topic 4
                                                  - Topic 5
                                                  10. Never fabricate information that is not supported by the retrieved context or reliable general knowledge.
                                                  11. If code is requested:
                                                  - Return complete, executable code.
                                                  - Explain important parts briefly.
                                                  - Preserve formatting.
                                                  12. If mathematical calculations are required:
                                                  - Show the calculation steps.
                                                  - Then provide the final answer.
                                                
                                                  Answer:""")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite",google_api_key=os.getenv("GOOGLE_API_KEY"),temperature=0,max_tokens=4096)
        # Convert retrieved documents into formatted text with source information
        # Convert retrieved documents into formatted text
    def format_docs(docs):
        formatted_text = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown Source")
            content = doc.page_content
            formatted_text.append(
                f"Source: {source}\n{content}"
        )
        return "\n\n".join(formatted_text)
# Create the RAG Chain
    rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)
# Chat Input
question = st.chat_input("Ask a question about your uploaded documents")

if question:
    with st.chat_message("user"):
        st.write(question)

    q = question.lower()

    compare_words = [
        "compare", "comparison", "difference", "different",
        "similar", "similarity", "vs", "versus", "distinguish"
    ]

    summary_words = [
        "summary", "summarize", "summarise",
        "overview", "brief", "abstract",
        "gist", "key points"
    ]

    
    # COMPARE DOCUMENTS
    if any(word in q for word in compare_words):

        from collections import defaultdict

        docs_by_file = defaultdict(list)

        # Group chunks by uploaded file
        for chunk in chunks:
            docs_by_file[chunk.metadata["source"]].append(chunk.page_content)

        context = ""

        for filename, pages in docs_by_file.items():
            context += f"\n\n= {filename} =\n"
            context += "\n".join(pages)

        compare_prompt = f"""
        You are an intelligent document comparison assistant.
        You are given multiple uploaded files.
        The files may include:
        - PDF
        - Word
        - PowerPoint
        - CSV
        - Excel
        - JSON
        - HTML
        - Source Code
        - Text files
        Rules:
        1. Compare EVERY uploaded file.
        2. Create ONE column for EACH uploaded file.
        3. Use the ORIGINAL uploaded filename as the column header.
        4. Never skip a file.
        5. Never repeat filenames.
        6. Never use temporary filenames like:
        /var/folders/...
        tmpxxxx.pdf
        tmpxxxx.csv
        If the file is a PDF, Word, PPT, Text, HTML etc., compare:
        - Title
        - Main Topic
        - Purpose
        - Key Concepts
        - Important Sections
        - Strengths
        - Weaknesses
        - Conclusion
        If the file is CSV or Excel, compare:
        - Dataset Name
        - Number of Rows
        - Number of Columns
        - Column Names
        - Data Types
        - Dataset Description
        - Possible Use Cases
        Return ONLY one Markdown table.
        Documents:
        {context}"""

        with st.spinner("Comparing documents..."):
            response = llm.invoke(compare_prompt).content
     
     # SUMMARIZE DOCUMENTS
    elif any(word in q for word in summary_words):

        from collections import defaultdict

        docs_by_file = defaultdict(list)

        for chunk in chunks:
            docs_by_file[chunk.metadata["source"]].append(chunk.page_content)

        context = ""

        for filename, pages in docs_by_file.items():
            context += f"\n\n= {filename} =\n"
            context += "\n".join(pages)

        summary_prompt = f"""
        You are an intelligent AI assistant.
        Summarize EVERY uploaded file separately.
        Rules:
        1. Use the filename as the heading.
        2. Do not merge different files.
        3. Summarize each file separately.
        4. Remove repetition.
        5. Use bullet points.
        6. Keep summaries concise.
        If the file is CSV or Excel include:
        - Dataset Name
        - Number of Rows
        - Number of Columns
        - Column Names
        - Dataset Description
        - Possible Applications
        Documents:
        {context}"""

        with st.spinner("Generating summary..."):
            response = llm.invoke(summary_prompt).content
    # NORMAL RAG
    else:

        with st.spinner("Searching documents..."):
            response = rag_chain.invoke(question)

    with st.chat_message("assistant"):
        st.write(response)