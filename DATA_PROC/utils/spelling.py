from transformers import pipeline

MAX_LENGTH = 256
corrector = pipeline("text2text-generation", model="bmd1905/vietnamese-correction-v2")

def correct_text(text: str):
    return corrector(text, max_length=MAX_LENGTH)

print(correct_text("Công ty đã up data 2 bức ảnh hẹn hò đều từ 2019. Kính mua mắc gì 5 6 năm sau không được saif, chưa kể trên ig của Kim Sae Ron có 1 tấm năm 2020 y chang kiểu tóc ảnh chụp chung kìa"))