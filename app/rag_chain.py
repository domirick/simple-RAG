import os

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers import RePhraseQueryRetriever

from vector_db import VectorDatabase

class RAGChain:
    def __init__(self):
        # Prepare arguments for VectorDatabase
        vector_db_args = {
            "vector_db_directory": os.environ.get("DB_DIRECTORY"),
            "documents_dir": os.environ.get("DOCUMENTS_DIR"),
            "embedding_model_name": os.environ.get("EMBEDDING_MODEL"),
            "reranker_model_name": os.environ.get("RERANKER"),
            "chunk_size": int(os.environ["CHUNK_SIZE"]),
            "chunk_overlap": int(os.environ["CHUNK_OVERLAP"]),
            "dense_top_k": int(os.environ["VECTOR_RETRIEVER_TOP_K"]) if "VECTOR_RETRIEVER_TOP_K" in os.environ else None,
            "sparse_top_k": int(os.environ["KEYWORD_RETRIEVER_TOP_K"]) if "KEYWORD_RETRIEVER_TOP_K" in os.environ else None,
            "reranker_top_k": int(os.environ["RERANKER_TOP_K"]) if "RERANKER_TOP_K" in os.environ else None,
            "glob": os.environ.get("DOCUMENTS_GLOB", None),
            "device": os.environ.get("DEVICE", None)
        }

        # Filter out None values to use VectorDatabase's defaults
        self.vector_db_args = {key: value for key, value in vector_db_args.items() if value is not None}

        # Index database & Create retriever
        vectorDB = VectorDatabase(**self.vector_db_args,
                                    recreate=False)
        retriever = vectorDB.get_retriever()

        # Create chain
        self._create_chain(retriever)
        print("RAG chain initialized.")

    def _create_chain(self, retriever):
        retriever_from_llm = RePhraseQueryRetriever.from_llm(
            retriever=retriever, llm=RAGChain.get_openai_client()
        )
        
        question_answer_chain = create_stuff_documents_chain(RAGChain.get_openai_client(), RAGChain.get_prompt())
        self.retrieval_chain = create_retrieval_chain(retriever_from_llm, question_answer_chain)

    def inference(self, chat_history):
        inference_result = self.retrieval_chain.invoke(
            {"input": RAGChain.format_history(chat_history)}
        )
        return RAGChain.format_answer(inference_result)

    def reindex_database(self):
        # Reindex the database
        vectordb = VectorDatabase(**self.vector_db_args,
                                    recreate=True)
        retriever = vectordb.get_retriever()

        # Recreate chain
        self._create_chain(retriever)
        print("Database reindexed.")

    def get_openai_client():
        # Initialize OpenAI client
        openai_client = ChatOpenAI(
            model=os.environ["LLM_MODEL"], 
            api_key=os.environ["OPENAI_API_KEY"], 
            base_url=os.environ["OPENAI_API_URL"],
            temperature=os.environ["TEMPERATURE"],
        )
        
        return openai_client
    
    def get_prompt():
        system_prompt = (
            "You are an intelligent assistant specialized in answering questions accurately and concisely, but in a friendly manner. "
            "Use the provided context to generate your answers. If the context does not contain sufficient "
            "information to answer the question, respond with 'I'm not sure based on the provided information.' "
            "Do not make up answers or provide information outside the given context. "
            "Focus on being clear, concise, and helpful."
            "Ensure that the language of your response matches the language of the user's question."
            "\n\n"
            "Context:\n{context}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        return prompt

    def format_history(chat_history):
        formatted_history = "\n".join([f"{message['role'].capitalize()}: {message['content']}" for message in chat_history])
        print(formatted_history)
        print("-"*100)
        return formatted_history
    
    def format_answer(unformatted_answer):
        formatted_answer = unformatted_answer['answer']
        formatted_answer += "\n\n"

        # Add sources
        formatted_answer += "📃:\n"

        # Create a set of unique sources from the context
        used_sources = {document.metadata['source'] for document in unformatted_answer['context']}

        # Trim the filename
        used_sources = [os.path.basename(source) for source in used_sources]

        # Form the answer
        for document in used_sources:
            formatted_answer += f"- {document}\n"
        return formatted_answer