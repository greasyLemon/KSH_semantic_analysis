import emoji
import re
from DATA_PROC.utils.abbreviation import replace_abbreviations

def clean_text(text: str):
    if not isinstance(text, str):  
        return ""

    text = re.sub(r"<a\s+[^>]*>(.*?)</a>", r"\1", text)  # Xóa thẻ <a>
    text = re.sub(r"https?://\S+|www\.\S+", "", text)  # Xóa link
    text = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", "", text)  # Xóa định dạng giờ
    text = re.sub(r"[ㅋㅎㅠㅜㄷ아ㄷ]+", "", text)  # Xóa ký tự tiếng Hàn
    text = replace_abbreviations(text)  # Thay thế từ viết tắt
    text = emoji.replace_emoji(text, replace="")  # Xóa emoji

    # Xóa @username
    text = re.sub(r"@\s*[^\s]+(?:\s+[^\s]+)*", "", text).strip()
    text = re.sub(r"^\s*@\s*$", "", text)  # Xóa @ nếu nó đứng một mình

    # Chỉ giữ lại chữ cái Latin (bao gồm tiếng Việt), số, dấu cách, dấu gạch dưới `_` và dấu `.,?!`
    text = re.sub(r"[^a-zA-Zàáảãạăắằẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợ"
                  r"ùúủũụưừứửữựỳýỷỹỵđ0-9_.,?!\s]", "", text)

    text = re.sub(r"\n+", " ", text)  # Xóa xuống dòng
    text = re.sub(r"\s+", " ", text).strip()  # Xóa khoảng trắng thừa
    return text.strip()
