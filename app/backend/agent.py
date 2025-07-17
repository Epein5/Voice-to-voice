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
                temperature=0.5,  # Balanced creativity
                top_p=0.8,
                top_k=30,
                max_tokens=60  # Limit to 1 sentence
            )

            prompt = (
                "तपाईं काठमाडौं विश्वविद्यालयको भर्ना सहायक हुनुहुन्छ। "
                "प्रयोगकर्तासँग मित्रवत् कुराकानी गर्नुहोस् र उनीहरूलाई भर्ना सम्बन्धी प्रश्न सोध्न प्रोत्साहन दिनुहोस्। "
                "केवल एक वाक्यमा मात्र छोटो जवाफ दिनुहोस्। "
                "यदि प्रयोगकर्ताले सामान्य कुराकानी गरेको छ भने उनीहरूलाई भर्ना, कोर्स, फिस, वा अध्ययन सम्बन्धी प्रश्न सोध्न भन्नुहोस्।\n\n"
                f"प्रयोगकर्ता: {text}\n"
                "तपाईं:"
            )

            response = llm.complete(prompt)
            return str(response).strip()

        except Exception as e:
            logging.error(f"Error in normal conversation: {e}")
            return "नमस्ते! म काठमाडौं विश्वविद्यालयको भर्ना सहायक हुँ। तपाईंले भर्ना, कोर्स, वा फिस सम्बन्धी प्रश्न सोध्न सक्नुहुन्छ।"