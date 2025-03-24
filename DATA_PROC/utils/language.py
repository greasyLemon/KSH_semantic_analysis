import fasttext

model_path = "DATA_PROC/utils/lid.176.bin"
model = fasttext.load_model(model_path)

def is_vietnamese(text: str):
    prediction = model.predict(text.replace("\n", " ").strip())[0][0]
    return prediction == "__label__vi"