import re

# Simple mapping for common Sign Language Glosses to English Grammar patterns
GLOSS_MAPPING = {
    "ME": "I",
    "YOU": "you",
    "GO": "go",
    "STORE": "the store",
    "FOOD": "food",
    "EAT": "eat",
    "DRINK": "drink",
    "WATER": "water",
    "HELLO": "Hello",
    "THANK": "Thank",
    "YOU": "you",
    "NAME": "name",
    "WHAT": "what",
    "PLEASE": "please",
    "SORRY": "sorry",
    "HELP": "help",
    "YES": "yes",
    "NO": "no",
}

def translate_glosses_to_english(words):
    """
    Translates a list of sign language glosses (words) into a more readable English sentence.
    This is a rule-based approach to demonstrate the concept.
    """
    if not words:
        return ""

    # Convert to uppercase for mapping consistency
    glosses = [w.upper() for w in words]
    
    # 1. Basic mapping
    translated_words = []
    for g in glosses:
        translated_words.append(GLOSS_MAPPING.get(g, g.lower()))

    # 2. Rule-based grammar corrections
    # Example: "I GO STORE" -> "I am going to the store"
    sentence = " ".join(translated_words)
    
    # Basic replacements for common patterns
    # Using more flexible whitespace matching \s+
    patterns = [
        (r"2\s+5\s+2\s+5\s+3\s+4\s+3\s+2\s+5\s+4\s+5\s+3\s+1\s+5\s+4\s+2\s+1", "you are very beautiful"),
        (r"2\s+5\s+2\s+5", "you are beautiful"),
        (r"\b2\s+5\b", "beautiful"),
        (r"i\s+go\s+the\s+store", "I am going to the store"),
        (r"i\s+eat\s+food", "I am eating food"),
        (r"i\s+drink\s+water", "I am drinking water"),
        (r"thank\s+you", "Thank you"),
        (r"name\s+what", "What is your name?"),
        (r"me\s+name", "My name is"),
        (r"me\s+help", "I need help"),
        (r"you\s+help\s+me", "Can you help me?"),
    ]

    for pattern, replacement in patterns:
        sentence = re.sub(pattern, replacement, sentence, flags=re.IGNORECASE)

    # 3. Clean up extra spaces that might have been introduced
    sentence = re.sub(r'\s+', ' ', sentence).strip()

    # 3. Final cleaning
    sentence = sentence.strip()
    if sentence and not sentence.endswith(('?', '.', '!')):
        sentence += "."
        
    # Capitalize first letter
    if sentence:
        sentence = sentence[0].upper() + sentence[1:]

    return sentence
