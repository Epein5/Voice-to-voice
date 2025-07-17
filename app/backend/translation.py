from llama_index.llms.gemini import Gemini
import os

def translate_nepali_to_english(text: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    llm = Gemini(model_name="models/gemini-2.5-flash")
    prompt = (
        "Translate the following Nepali text to English. Answer in one short, direct sentence. Only return the answer, no explanations, no repetition.\n\n"
        f"{text}"
    )
    result = llm.complete(prompt)
    return str(result).strip()

def translate_english_to_nepali(text: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    llm = Gemini(model_name="models/gemini-2.5-flash")
    prompt = (
        "Translate the following English text to Nepali for a phone call. "
        "Answer in one short, direct sentence. Only return the answer, no explanations, no repetition. "
        "Do not use any digits in the Nepali response; write numbers in Nepali words. "
        "Add spaces between some in Nepali words and between letters in acronyms (like ए आई for एआई) for better TTS pauses. "
        "\n\nEnglish response: " + text
    )
    result = llm.complete(prompt)
    return str(result).strip() 