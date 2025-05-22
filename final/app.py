import os
from data_loader import load_and_preprocess
from qa_system import QASystem
from speech_utils import listen_to_user, speak_text

WAKE_WORDS = ["hello assistant", "hi assistant", "hey assistant", "helloassistent"]

def main():
    print("Loading files for analysis...")

    file_paths = [
        "inventory.csv",
        "report.pdf",
        "notes.docx",
        "summary.txt"
    ]

    file_paths = [fp for fp in file_paths if os.path.exists(fp)]
    if not file_paths:
        print("No valid files found. Please check your file paths.")
        return

    texts = load_and_preprocess(file_paths)
    print(f"✅ Loaded and processed {len(texts)} text chunks from {len(file_paths)} file(s).")

    qa = QASystem(texts, device="cpu")

    print("\n🎤 Say a wake word like 'Hello Assistant' to start asking questions.")
    print("🔇 Say 'exit' to stop answering but keep the program running.")
    print("❌ Say 'goodbye' to quit the program.\n")

    running = True
    active = False

    while running:
        print("🔎 Waiting for wake word...")
        user_input = listen_to_user()
        if user_input is None:
            continue

        lower_input = user_input.lower().strip()

        if lower_input == "goodbye":
            print("👋 Goodbye! Closing the program.")
            speak_text("Goodbye!")
            running = False

        elif lower_input in WAKE_WORDS:
            speak_text("Hello! I’m your assistant. What can I help you with today?")
            active = True

            while active:
                print("🎧 Listening for your question...")
                question = listen_to_user()
                if question is None:
                    continue

                q_lower = question.lower().strip()
                if q_lower == "exit":
                    speak_text("Okay, I’ll stop answering for now. Say 'hello assistant' again when you're ready.")
                    active = False
                    break
                elif q_lower == "goodbye":
                    speak_text("Goodbye!")
                    running = False
                    active = False
                    break

                answer = qa.answer(question)
                print("🤖 Bot:", answer)
                speak_text(answer)

if __name__ == "__main__":
    main()
