# memories.py
from datetime import datetime
from model import llm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from prompt import Prompts

# Initialize prompts
prompts = Prompts()

# Embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_user_vector_store():
    """
    Get a user-specific Chroma vector store with LangChain wrapper.
    """
    persist_path = "./chroma_db"
    collection_name = f"personal_chat_memories"

    # ✅ Use LangChain's Chroma wrapper directly
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=persist_path
    )

    return vector_store

class Memories:
    """
    Handles retrieval, storage, and decision-making for user-specific conversational memories.
    """

    def __init__(self):
        self.vector_store = get_user_vector_store()

    def retrieve_memories(self, query: str, k: int = 3) -> str:
        """
        Retrieve top-k relevant memories from the vector DB for a given query.
        """
        memories = self.vector_store.similarity_search(query, k=k)

        if memories:
            print(f"[LOG] Retrieved {len(memories)} relevant memories")
            for idx, doc in enumerate(memories, start=1):
                print(f" Memory {idx}: {doc.page_content[:100]}...")
        else:
            print(f"[LOG] No prior context found.")

        return "\n".join(doc.page_content for doc in memories) if memories else "No prior context."

    def store_memory(self, user_input: str, assistant_response: str) -> None:
        """
        Store new memory in the vector DB with a timestamp.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_memory_text = f"[{timestamp}]\nUser: {user_input}\nMe: {assistant_response}"

        self.vector_store.add_documents([Document(page_content=new_memory_text)])

        # print(f"[LOG] New memory stored:\n{new_memory_text}")

    def decide_to_memorize(self, user_input: str, assistant_response: str) -> bool:
        """
        Ask the LLM whether the exchange should be remembered.
        Returns True if LLM says YES, False otherwise.
        """
        decision_prompt = prompts.get_memory_decision_prompt().invoke({
            "user_input": user_input,
            "assistant_response": assistant_response
        })

        # print(f"\n[LOG] Decision Prompt Sent to LLM:\n{decision_prompt}")

        decision = llm.invoke(decision_prompt)
        decision_text = decision.content.strip().upper()

        print(f"[LOG] LLM Decision Response: {decision_text}")

        return "YES" in decision_text
