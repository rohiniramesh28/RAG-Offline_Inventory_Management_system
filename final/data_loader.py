import os
import pandas as pd
from langchain.document_loaders import (
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_preprocess(file_paths):
    all_texts = []

    # Mapping file extensions to loaders
    loaders = {
        '.pdf': UnstructuredPDFLoader,
        '.docx': UnstructuredWordDocumentLoader,
        '.txt': TextLoader
    }

    for file_path in file_paths:
        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.csv':
            df = pd.read_csv(file_path)

            def row_to_text(row):
                return ". ".join(
                    [f"The {col} is {val}" for col, val in row.items() if pd.notnull(val)]
                ) + "."

            all_texts.extend(df.apply(row_to_text, axis=1).tolist())

        elif ext in loaders:
            loader_class = loaders[ext]
            loader = loader_class(file_path)
            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_documents(documents)
            all_texts.extend([chunk.page_content for chunk in chunks])

        else:
            print(f"Unsupported file format: {file_path}")

    return all_texts
