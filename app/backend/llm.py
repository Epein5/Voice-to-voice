from llama_index.llms.gemini import Gemini
import os
import logging

def summarize_rag_to_nepali(context: str, query: str) -> str:
    if not context:
        return "माफ गर्नुहोस्, थप जानकारी उपलब्ध छैन।"
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logging.error("GEMINI_API_KEY not found in environment")
        return "माफ गर्नुहोस्, थप जानकारी उपलब्ध छैन।"
    llm = Gemini(
        model_name="models/gemini-2.5-flash",
        temperature=0.3,
        top_p=0.8,
        top_k=20,
        max_tokens=50  # Limit to 1 sentence
    )
    prompt = (
        "तपाईं काठमाडौं विश्वविद्यालयको भर्ना सल्लाहकार हुनुहुन्छ। "
        "तलको सन्दर्भको आधारमा प्रयोगकर्ताको प्रश्नको उत्तर दिनुहोस्। "
        "केवल एक वाक्यमा मात्र छोटो र स्पष्ट जवाफ दिनुहोस्। "
        "संख्याहरूलाई नेपाली शब्दमा लेख्नुहोस्। "
        "सीधा र संक्षिप्त भएर जवाफ दिनुहोस्।\n\n"
        f"प्रश्न: {query}\n"
        f"जानकारी:\n{context}\n\n"
        "उत्तर:"
    )
    try:
        response = llm.complete(prompt)
        return str(response)
    except Exception as e:
        logging.error(f"Error in Gemini LLM summarization: {e}")
        return "माफ गर्नुहोस्, थप जानकारी उपलब्ध छैन।" 