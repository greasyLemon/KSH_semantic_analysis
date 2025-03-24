import json
import re

abbreviations_path = "DATA_PROC/utils/abbreviations.json"

with open(abbreviations_path, "r", encoding="utf-8") as f:
    abbreviations = json.load(f)

abbr_regex = re.compile(r'\b(' + '|'.join(re.escape(k) for k in abbreviations.keys()) + r')\b', re.IGNORECASE)

def replace_abbreviations(text):
    def replace_match(match):
        word = match.group(0)
        replacement = abbreviations[word.lower()]
        return replacement.capitalize() if word[0].isupper() else replacement

    return abbr_regex.sub(replace_match, text)
