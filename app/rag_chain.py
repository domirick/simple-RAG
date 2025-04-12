import os

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

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
        question_answer_chain = create_stuff_documents_chain(RAGChain.get_openai_client(), RAGChain.get_prompt())
        self.retrieval_chain = create_retrieval_chain(retriever, question_answer_chain)
        print("RAG chain initialized.")

    #def inference(self, query, chat_history=None):
    #    input_with_history = f"{RAGChain.format_history(chat_history)}\nUser: {query}"
    #    inference_result = self.retrieval_chain.invoke(
    #        {"input": input_with_history}
    #    )
    #    return RAGChain.format_answer(inference_result)

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
        question_answer_chain = create_stuff_documents_chain(RAGChain.get_openai_client(), RAGChain.get_prompt())
        self.retrieval_chain = create_retrieval_chain(retriever, question_answer_chain)
        print("Database reindexed.")

    def get_openai_client():
        # Initialize OpenAI client
        openai_client = ChatOpenAI(
            model=os.environ["LLM_MODEL"], 
            api_key=os.environ["OPENAI_API_KEY"], 
            base_url=os.environ["OPENAI_API_URL"],
            temperature=0.7,
        )
        
        return openai_client
    
    def get_prompt():
        system_prompt = (
            "You are an intelligent assistant specialized in answering questions accurately and concisely, but in a friendly manner. "
            "Use the provided context to generate your answers. If the context does not contain sufficient "
            "information to answer the question, respond with 'I'm not sure based on the provided information.' "
            "Do not make up answers or provide information outside the given context. "
            "Focus on being clear, concise, and helpful."
            "Answer language should match the question language."
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
        formatted_answer += "📃:\n"
        used_sources = {document.metadata['source'] for document in unformatted_answer['context']}
        for document in used_sources:
            formatted_answer += f"- {document}"
        #for document in unformatted_answer['context']:
        #    formatted_answer += f"- {document.metadata['source']}"
        return formatted_answer