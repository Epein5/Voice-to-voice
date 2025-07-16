"""
RAGHandler: Handles Retrieval-Augmented Generation operations
"""
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
import chromadb
import os
import logging
from pathlib import Path

class RAGHandler:
    def get_nepali_answer(self, query: str):
        """
        Retrieves context for the query and returns only the Nepali answer for TTS.
        Use this method to get the final output for TTS, not the raw context.
        """
        context = self.query(query)
        nepali_answer = self.summarize_in_nepali(context, query)
        # Only return the Nepali answer, not the context
        return nepali_answer
    def summarize_in_nepali(self, context: str, query: str):
        """
        Passes the retrieved context and user query to Gemini LLM with a proper RAG prompt for Nepali summarization.
        """
        if not context:
            return "माफ गर्नुहोस्, थप जानकारी उपलब्ध छैन।"
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logging.error("GEMINI_API_KEY not found in environment")
            return "माफ गर्नुहोस्, थप जानकारी उपलब्ध छैन।"
        llm = Gemini(
            model_name="models/gemini-2.5-flash",
            temperature=0.1,
            max_output_tokens=256,
            top_p=0.8,
            top_k=20
        )
        prompt = (
            "तपाईंलाई तलको सन्दर्भ र प्रयोगकर्ताको प्रश्न दिइएको छ। "
            "सन्दर्भको आधारमा छोटो र संक्षिप्त रूपमा नेपालीमा उत्तर दिनुहोस्। "
            "कृपया अंकहरूलाई शब्दमा लेख्नुहोस्, संख्यात्मक अंक प्रयोग नगर्नुहोस्।\n"
            f"\nप्रश्न: {query}\n"
            f"सन्दर्भ:\n{context}"
        )
        try:
            response = llm.complete(prompt)
            return str(response)
        except Exception as e:
            logging.error(f"Error in Gemini LLM summarization: {e}")
            return "माफ गर्नुहोस्, थप जानकारी उपलब्ध छैन।"
    def __init__(self):
        self.retriever = None
        self._initialize()

    def _initialize(self):
        try:
            embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
            Settings.embed_model = embed_model
            vector_db_path = Path("/media/epein5/Data1/Voice-to-voice/vector-db/")
            if not vector_db_path.exists():
                logging.warning(f"Vector database not found at {vector_db_path}. RAG will be limited.")
                self.retriever = None
                return
            chroma_client = chromadb.PersistentClient(path="/media/epein5/Data1/Voice-to-voice/vector-db/")
            collections = chroma_client.list_collections()
            if not collections:
                logging.warning("No collections found in ChromaDB. RAG will be limited.")
                self.retriever = None
                return
            chroma_collection = None
            for col in collections:
                if getattr(col, 'name', None) == 'kusoe_rag_llm':
                    chroma_collection = col
                    break
            if chroma_collection is None:
                logging.warning("'kusoe_rag_llm' collection not found in ChromaDB. RAG will be limited.")
                self.retriever = None
                return
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_context
            )
            self.retriever = index.as_retriever(similarity_top_k=3)
        except Exception as e:
            logging.error(f"Error initializing RAG system: {e}")
            self.retriever = None

    def query(self, text: str):
        logging.info(f"RAGHandler.query called with text: {text}")
        if not self.retriever:
            logging.error("Retriever not initialized. Returning fallback.")
            return None
        try:
            nodes = self.retriever.retrieve(text)
            context_chunks = []
            for node in nodes:
                if hasattr(node, 'text'):
                    context_chunks.append(node.text)
            context = '\n---\n'.join(context_chunks) if context_chunks else None
            logging.info(f"RAG retrieved context chunks:\n{context}")
            return context
        except Exception as e:
            logging.error(f"Error in RAG retrieval: {e}")
            return None
