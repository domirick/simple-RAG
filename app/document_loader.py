from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader

def load_documents(directory: str, chunk_size: int, chunk_overlap: int, glob: str = "**/[!.]*") -> list:
    """
    Load documents from a directory using a glob pattern, then splitting them into chunks.
    Args:
        directory (str): The directory to load documents from.
        glob (str): The glob pattern to match files. Defaults to "**/[!.]*".

    Returns:
        list: A list of document chunks.
    """
    loader = DirectoryLoader(path=directory, glob=glob)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)

    return chunks