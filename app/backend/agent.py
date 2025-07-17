from llama_index.llms.gemini import Gemini
import os
import logging

class Agent:
    def __init__(self):
        # More specific RAG keywords - only for specific information queries
        self.rag_keywords = [
            "फिस", "शुल्क", "पैसा", "रुपैयाँ", "कति", "कोर्स", "भर्ना", "प्रवेश", "परीक्षा", "सिलेबस",
            "विषय", "समय", "अवधि", "वर्ष", "सेमेस्टर", "कलेज", "विश्वविद्यालय", "डिग्री", "सर्टिफिकेट",
            "fee", "cost", "price", "course", "admission", "exam", "syllabus", "duration", "college", "university"
        ]

    def classify(self, text: str) -> str:
        # More selective classification - only trigger RAG for specific information queries
        lowered = text.lower()

        # Check if it's a specific information query
        rag_indicators = 0
        for kw in self.rag_keywords:
            if kw in lowered:
                rag_indicators += 1

        # Only use RAG if there are clear indicators of information seeking
        if rag_indicators >= 1 and any(q in lowered for q in ["कति", "के", "कसरी", "कहाँ", "what", "how", "where", "when"]):
            return "rag"

        return "normal"

    def normal_conversation(self, text: str) -> str:
        """Handle normal conversation using Gemini LLM"""
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                logging.error("GEMINI_API_KEY not found in environment")
                return "माफ गर्नुहोस्, म अहिले कुराकानी गर्न सक्दिन।"

            llm = Gemini(
                model_name="models/gemini-2.5-flash",
                temperature=0.7,  # More creative for conversation
                top_p=0.9,
                top_k=40
            )

            prompt = (
                "तपाईं एक मित्रवत् र सहयोगी नेपाली सहायक हुनुहुन्छ। "
                "प्रयोगकर्तासँग प्राकृतिक र न्यानो कुराकानी गर्नुहोस्। "
                "२-३ वाक्यमा जवाफ दिनुहोस्। "
                "औपचारिक नभएर मित्रवत् भएर कुरा गर्नुहोस्। "
                "यदि प्रयोगकर्ताले कुनै विशेष जानकारी सोधेको छैन भने सामान्य कुराकानी गर्नुहोस्।\n\n"
                f"प्रयोगकर्ता: {text}\n"
                "तपाईं:"
            )

            response = llm.complete(prompt)
            return str(response).strip()

        except Exception as e:
            logging.error(f"Error in normal conversation: {e}")
            return "नमस्ते! म तपाईंसँग कुराकानी गर्न खुसी छु। तपाईंलाई कसरी सहयोग गर्न सक्छु?"