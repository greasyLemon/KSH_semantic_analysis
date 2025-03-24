import pandas as pd
import os

def xlsx_to_csv(folder_path, all_comments):
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.xlsx'):
            file_path = os.path.join(folder_path, file_name)
            
            df = pd.read_excel(file_path, skiprows=14)
            
            all_comments = pd.concat([all_comments, df['Comment Text']], ignore_index=True)

    all_comments.drop_duplicates(inplace=True)
    all_comments.dropna(inplace=True)
    all_comments.reset_index(drop=True, inplace=True)
    all_comments.columns = ["Comment Text"]
    all_comments.index.name = "ID"

    all_comments.to_csv('DATA/comments.csv', index=True)

    print("Đã trích xuất và lưu tất cả các bình luận không trùng lặp vào file 'comments.csv' với index.")

def main():
    folder_path = 'DATA/raw'
    all_comments = pd.DataFrame()

    xlsx_to_csv(folder_path=folder_path, all_comments=all_comments)

if __name__ == "__main__":
    main()