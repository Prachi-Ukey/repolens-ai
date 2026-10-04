import sys
import os

user_site = os.path.expanduser('~/AppData/Roaming/Python/Python313/site-packages')
if os.path.exists(user_site) and user_site not in sys.path:
    sys.path.insert(0, user_site)

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ai.rag import RAGEngine
from app.config import settings

TEST_QUESTIONS = [
    ("Test 1", "Where is authentication implemented and how does the login process work?"),
    ("Test 2", "How is the user's password verified?"),
    ("Test 3", "How does the application protect the dashboard from unauthenticated users?"),
    ("Test 4", "Where is the database connection configured?"),
    ("Test 5", "How does vehicle damage detection work?"),
    ("Test 6", "Where is the YOLO model loaded?"),
    ("Test 7", "How is the insurance cost estimated?")
]

def run_all_rag_tests():
    repo_id = "b41c69a2-e261-4f26-8fd4-5208ce869216"
    print("==================================================================")
    print(f"RUNNING REPOLENS RAG TEST SUITE")
    print(f"Repository ID: {repo_id}")
    print(f"Active LLM Provider: {settings.LLM_PROVIDER}")
    print(f"Active Embedding Provider: {settings.EMBEDDING_PROVIDER}")
    print("==================================================================\n")

    engine = RAGEngine()

    for label, question in TEST_QUESTIONS:
        print(f"------------------------------------------------------------------")
        print(f"[{label}] {question}")
        print(f"------------------------------------------------------------------")

        answer, sources, grounded = engine.ask_question(repo_id, question, top_k=5)

        print("\nCITED SOURCES:")
        for s in sources:
            print(f"  - {s.file_path}: L{s.start_line}-L{s.end_line} ({s.symbol_name or 'Scope'})")

        print("\nGENERATED GROUNDED ANSWER:")
        print(answer)
        print("\n")

if __name__ == "__main__":
    run_all_rag_tests()
