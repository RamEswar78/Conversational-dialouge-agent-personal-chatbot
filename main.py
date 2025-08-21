# main.py
from dotenv import load_dotenv
import os
from model import llm  # ✅ Your existing LLM setup
from prompt import Prompts
from memory import Memories

# ✅ Load environment variables
load_dotenv()


def chat_loop():
    """Main chat loop that interacts with the user."""
    memories_inst = Memories()
    prompts = Prompts()

    while True:
        # 🟢 Get user input
        user_input = input("\nYou: ")

        # 🔴 Exit condition
        if "exit" in user_input.lower():
            print("Exiting the chat. Goodbye!")
            break

        # 1️⃣ Retrieve relevant memories
        memory_content = memories_inst.retrieve_memories(user_input)

        # 2️⃣ Prepare personalized AI prompt
        personal_ai = prompts.get_personal_ai_prompt().invoke({
            "memories": memory_content,
            "user_input": user_input
        })

        # print(f"[DEBUG] Type of personal_ai: {type(personal_ai)}")

        # 3️⃣ Get LLM response
        response = llm.invoke(personal_ai)
        print("Friend:", response.content)

        # 4️⃣ Ask agent if this should be remembered
        if memories_inst.decide_to_memorize(user_input, response.content):
            memories_inst.store_memory(user_input, response.content)
            print("[LOG] Memory stored ✅")
        else:
            print("[LOG] Memory not stored (LLM decision: NO).")


if __name__ == "__main__":
    chat_loop()
