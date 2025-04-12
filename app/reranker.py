from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

class Reranker:
    def get_retriever(model_name: str, retriever, top_k: int = 5) -> ContextualCompressionRetriever:
        """
        Creates a reranker retriever.
        Args:
            model_name (str): The name of the model to use for reranking from HuggingFaceCrossEncoder.
            retriever: The base retriever to use for initial retrieval.
            top_k (int): The number of top results to retrieve after reranking. Defaults to 5.
        Returns:
            ContextualCompressionRetriever: The compression retriever.
        """
        model = HuggingFaceCrossEncoder(model_name=model_name)
        compressor = CrossEncoderReranker(model=model, top_n=top_k)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=retriever
        )
        return compression_retriever