import emoji
import re
from DATA_PROC.utils.abbreviation import replace_abbreviations

def clean_text(text: str):
    if not isinstance(text, str):  # Kiểm tra nếu text không phải chuỗi
        return ""
    # Xóa các thẻ <a>...</a>
    text = re.sub(r"<a\s+[^>]*>(.*?)</a>", r"\1", text) 
    
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    text = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", "", text)

    text = re.sub(r"[ㅋㅎㅠㅜㄷ아ㄷ]+", "", text)

    text = replace_abbreviations(text)

    text = emoji.replace_emoji(text, replace="")  # Xóa emoji

    # Xóa các tag @ (bao gồm cả dấu cách)
    text = re.sub(r"@\s*[^\s]+(?:\s+[^\s]+)*", "", text).strip()
    
    # Xóa @ nếu nó đứng một mình
    text = re.sub(r"^\s*@\s*$", "", text)

    # Xóa các ký tự đặc biệt như: ♡ ω ♡, ♪, ~, ★, ♥, ☆
    text = re.sub(r"[♡ω♪★♥☆~()]", "", text)

    # Xóa mặt cười như :)), =))), ^^, :3, :v
    text = re.sub(r"[:;=8xX]-?[)DpP3vV]+|[\^\*]+", "", text)

    text = re.sub(r"\n+", " ", text)

    # Xóa khoảng trắng thừa
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip()