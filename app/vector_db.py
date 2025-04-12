import os

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.retrievers import EnsembleRetriever
from langchain.docstore.document import Document

from document_loader import load_documents
from keyword_search import KeywordRetriever
from reranker import Reranker


class VectorDatabase:
    def __init__(self, vector_db_directory: str, documents_dir: str, embedding_model_name: str, reranker_model_name: str, chunk_size: int, chunk_overlap: int, dense_top_k: int = 5, sparse_top_k: int = 5, reranker_top_k: int = 5, glob: str="**/[!.]*", device: str = "cpu", recreate: bool = False):
        """
        Initialize the VectorDB class.
        Args:            
            vector_db_directory (str): The directory which would contain the vector database.
            documents_dir (str): The directory containing the documents to be indexed.
            embedding_model_name (str): The name of the embedding model to use from HuggingFaceEmbeddings.
            reranker_model_name (str): The name of the reranker model to use from HuggingFaceCrossEncoder.
            chunk_size (int): The size of the chunks to split the documents into.
            chunk_overlap (int): The overlap between chunks.
            dense_top_k (int): The number of top results to retrieve using dense retrieval. Defaults to 5.
            sparse_top_k (int): The number of top results to retrieve using sparse retrieval. Defaults to 5.
            reranker_top_k (int): The number of top results to retrieve after reranking. Defaults to 5.
            glob (str): The glob pattern to match files. Defaults to "**/[!.]*".
            device (str): The device to use for the embedding model ("cpu" or "cuda"). Defaults to "cpu".
            recreate (bool): Whether to recreate the vector database if it already exists. Defaults to False.
        """
        # Init vector database
        if recreate:
            # Delete the existing vector db
            if VectorDatabase.vector_db_exists(vector_db_directory):
                try:
                    for root, dirs, files in os.walk(vector_db_directory, topdown=False):
                        for name in files:
                            os.remove(os.path.join(root, name))
                        for name in dirs:
                            os.rmdir(os.path.join(root, name))
                    os.rmdir(vector_db_directory)
                except Exception as e:
                    raise Exception(f"Error deleting vector db in {vector_db_directory} with {e}")
        
        
        # Load documents - it is always required for the keyword retriever
        print(f"Loading documents from {documents_dir} with glob pattern {glob}")
        chunks = load_documents(documents_dir, chunk_size, chunk_overlap, glob)

        # Check if the vector db exists
        if VectorDatabase.vector_db_exists(vector_db_directory):
            # Load vector db
            print("Loading vector db...")
            self.vectorstore = VectorDatabase.load_vector_db(
                directory=vector_db_directory,
                embedding_model_name=embedding_model_name,
                device=device
            )
        else:
            # Create vector db
            print("Creating vector db...")
            self.vectorstore = VectorDatabase.create_vector_db(
                directory=vector_db_directory,
                embedding_model_name=embedding_model_name,
                chunks=chunks,
                device=device
            )
        
        # Create a retriever from the vectorstore
        self.vector_retriever = self.vectorstore.as_retriever(search_kwargs={"n_results": dense_top_k})

        # Init keyword search retriever
        print("Initializing keyword retriever...")
        self.keyword_retriever = KeywordRetriever.get_retriever(
            chunks=chunks,
            top_k=sparse_top_k
        )

        # Combine the two retrievers with ensemble
        print("Initializing ensemble retriever...")
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.keyword_retriever, self.vector_retriever], weights=[0.5, 0.5]
        )

        # Set up reranker
        print("Initializing the reranker...")
        self.compression_retriever = Reranker.get_retriever(
            model_name=reranker_model_name,
            retriever=self.ensemble_retriever,
            top_k=reranker_top_k
        )

    def vector_db_exists(directory: str) -> bool:
        """
        Check if the vector database exists in the specified directory.
        Args:
            directory (str): The directory to check for the vector database.
        Returns:
            bool: True if the vector database exists, False otherwise.
        """
        return os.path.exists(directory) and os.path.isdir(directory) and len(os.listdir(directory)) != 0
    
    def create_vector_db(directory: str, embedding_model_name: str, chunks: list[Document], device: str = "cpu"):
        """
        Create a vector database from the given directory.
        Args:
            directory (str): The directory containing the documents to be indexed.
            embedding_model (str): The name of the embedding model to use from HuggingFaceEmbeddings.
            chunks (list[Document]): The list of document chunks to be indexed.
            device (str): The device to use for the embedding model ("cpu" or "cuda"). Defaults to "cpu".
        Returns:
            FAISS: The created vectorstore object.
        """

        # Create embeddings using HuggingFaceEmbeddings
        embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs={"device": device})

        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=embedding_model
        )

        # Save the vectorstore to disk
        VectorDatabase.save_vector_db(
            vectorstore=vectorstore,
            directory=directory
        )

        return vectorstore

    def load_vector_db(directory: str, embedding_model_name: str, device:str = "cpu") -> FAISS:
        """
        Load the vector database from the specified directory.
        Args:
            directory (str): The directory containing the vector database.
            embedding_model_name (str): The name of the embedding model to use from HuggingFaceEmbeddings.
            device (str): The device to use for the embedding model ("cpu" or "cuda"). Defaults to "cpu".
        Returns:
            FAISS: The loaded vectorstore object.
        """

        # Create embeddings using HuggingFaceEmbeddings
        embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs={"device": device})

        # Load the vectorstore from the specified directory
        vectorstore = FAISS.load_local(
            folder_path=directory,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True,
        )

        return vectorstore

    def save_vector_db(vectorstore: FAISS, directory: str):
        """
        Save the vector database to the specified directory.
        Args:
            vectorstore (FAISS): The vectorstore object to save.
            directory (str): The directory to save the vector database to.
        """
        vectorstore.save_local(directory)

    def get_retriever(self):
        """
        Get the reranker retriever.
        """
        return self.compression_retriever

    # Query functions

    def query(self, query_text: str) -> str:
        """
        Hybrid search with reranking.
        This function first retrieves documents using the ensemble retriever and then reranks them using the compression retriever.
        Args:
            query_text (str): The query text to search for.
        Returns:
            str: The concatenated page content of the retrieved documents.
        """
        retrieved_docs = self.compression_retriever.invoke(query_text)
        return "\n".join([doc.page_content for doc in retrieved_docs])
    
    def query_docs(self, query_text: str) -> list[Document]:
        """
        Hybrid search with reranking.
        This function first retrieves documents using the ensemble retriever and then reranks them using the compression retriever.
        Args:
            query_text (str): The query text to search for.
        Returns:
            list[Document]: A list of retrieved documents.
        """
        retrieved_docs = self.compression_retriever.invoke(query_text)
        return retrieved_docs
    

    def query_without_reranker(self, query_text: str) -> list[Document]:
        """
        Hybrid search without reranking.
        This function retrieves documents using the ensemble retriever.
        Args:
            query_text (str): The query text to search for.
        Returns:
            list[Document]: A list of retrieved documents.
        """
        return self.ensemble_retriever.invoke(query_text)

    def dense_search(self, query_text: str, top_k: int) -> list[Document]:
        """
        Query the vectorstore and return the raw documents.
        Args:
            query_text (str): The query text to search for in the vectorstore.
            top_k (int): The number of top results to return.
        Returns:
            list[Document]: A list of retrieved documents.
        """
        return self.vectorstore.similarity_search(query_text, k=top_k)

    def sparse_search(self, query_text: str) -> list[Document]:
        """
        Query the keyword retriever and return the raw documents.
        Args:
            query_text (str): The query text to search for in the keyword retriever.
        Returns:
            list[Document]: A list of retrieved documents.
        """
        return self.keyword_retriever.invoke(query_text)