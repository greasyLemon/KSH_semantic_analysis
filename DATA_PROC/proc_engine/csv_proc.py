import pandas as pd
from DATA_PROC.utils.tokenizer import get_token_num
from DATA_PROC.utils.language import is_vietnamese
from DATA_PROC.utils.cleaning import clean_text

# run in Terminal: python -m DATA_PROC.proc_engine.csv_proc
def main():
    comment_path = "DATA/comments3.csv"
    df = pd.read_csv(comment_path, index_col="ID")

    df.drop(columns=["Comment Text"], inplace=True)
    df.rename(columns={"Translated": "Comment Text"}, inplace=True)
    
    df["Comment Text"] = df["Comment Text"].astype(str).apply(clean_text)

    df["Comment Text"] = df["Comment Text"].replace(r'^\s*$', pd.NA, regex=True)
    df.dropna(inplace=True)

    df["Token Count"] = df["Comment Text"].apply(get_token_num)

    filtered_out_comments = df[df["Token Count"] < 3]
    df = df[(df["Token Count"] >= 3) & (df["Token Count"] <= 250)]

    df = df[df["Comment Text"].apply(is_vietnamese)]

    df.reset_index(drop=True, inplace=True)
    filtered_out_comments.reset_index(drop=True, inplace=True)

    df.to_csv("DATA/processed_comments3.csv", index=True, index_label="ID")
    filtered_out_comments.to_csv("DATA/filtered_out_comments3.csv", index=True, index_label="ID")

    print("Đã lọc và xuất file bình luận hợp lệ và bình luận ngắn.")

if __name__ == "__main__":
    main()
