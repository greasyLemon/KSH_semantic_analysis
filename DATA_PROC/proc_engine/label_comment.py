import os
import pandas as pd
import time
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
# Chạy trên vscode lâu thì copy code chạy trong colab
filtered_comment_path = "DATA/processed_comments.csv"
labeled_comment_path = "DATA/labeled_comments.csv"

api_keys_str = os.getenv("GEMINI_API_KEY", "[]")
api_keys = json.loads(api_keys_str)

current_key_index = 0

client = genai.Client(api_key=api_keys[current_key_index])

df = pd.read_csv(filtered_comment_path)
comments = df["Comment Text"].to_list()

sys_instruct = (
    "Bạn là một người gán nhãn cảm xúc cho các bình luận liên quan đến tranh cãi hiện tại của nam diễn viên Kim Soo Hyun. "
    "Tranh cãi này bắt đầu khi có thông tin về mối quan hệ giữa Kim Soo Hyun và cố diễn viên Kim Sae Ron, với cáo buộc rằng họ đã hẹn hò khi cô còn ở tuổi vị thành niên. "
    "Vụ việc đã thu hút sự chú ý lớn từ báo chí, dẫn đến nhiều bài viết và phân tích về hành vi của cả hai. "
    "Gia đình của Kim Sae Ron cũng đã lên tiếng, bày tỏ sự thất vọng và lo lắng về tình huống này. "
    "Đáp lại, Kim Soo Hyun đã tổ chức một buổi họp báo, trong đó anh phủ nhận các cáo buộc và nhấn mạnh rằng mối quan hệ của họ chỉ là đồng nghiệp. "
    "Tuy nhiên, Dispatch đã xác nhận rằng Kim Soo Hyun và Kim Sae Ron từng có mối quan hệ tình cảm bí mật. "
    "Ngoài ra, Dispatch cũng tiết lộ rằng trước khi qua đời, Kim Sae Ron đã gửi tin nhắn cầu cứu Kim Soo Hyun nhưng không nhận được phản hồi. "
    "Về vấn đề tài chính, công ty quản lý Gold Medalist của Kim Soo Hyun đã gửi giấy báo nợ cho Kim Sae Ron, yêu cầu cô trả lại số tiền 700 triệu won mà công ty đã thanh toán thay cô trước đó. "
    "Viện Garosero tiếp tục gây tranh cãi khi công bố video riêng tư của Kim Sae Ron và tiết lộ tin nhắn Telegram giữa cô và Kim Soo Hyun. "
    "Những thông tin này đã làm dấy lên nhiều tranh cãi và áp lực đối với Kim Soo Hyun, dẫn đến việc nhiều nhãn hàng quyết định chấm dứt hợp đồng với nam diễn viên. "
    "Hãy đọc từng bình luận dưới đây và xác định cảm xúc chính của bình luận đó, chỉ trả về một nhãn chính duy nhất với các nhãn chính là: Ủng hộ Kim Soo Hyun, Không ủng hộ Kim Soo Hyun, Trung lập, Không liên quan. "
    "Nhãn 'Không liên quan' chỉ được sử dụng cho những bình luận thực sự vô nghĩa hoặc không liên quan đến tranh cãi trên. "
    "Nhãn 'Trung lập' dành cho những bình luận không thể hiện rõ quan điểm của người viết, vẫn đang tìm hiểu về sự việc, hoặc không vội kết luận khi chưa biết hết thông tin."
)

def switch_api_key():
    global current_key_index, client
    current_key_index = (current_key_index + 1) % len(api_keys)
    client = genai.Client(api_key=api_keys[current_key_index])
    print(f"Đổi sang API key thứ {current_key_index + 1}")

def send_request_with_retry(comment, retries=3, delay=2):
    for i in range(retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                config=types.GenerateContentConfig(
                    response_mime_type='text/x.enum',
                    response_schema={
                        "type": "STRING",
                        "enum": ["Ủng hộ", "Không ủng hộ", "Trung lập", "Không liên quan"],
                    },
                    system_instruction=sys_instruct
                ),
                contents=[{"text": comment}]
            )
            return response.text
        except Exception as e:
            print(f"Lỗi: {e}. Thử lại lần {i + 1}...")
            if "RESOURCE_EXHAUSTED" in str(e):
                print("Quota API đã hết, đổi sang key khác...")
                switch_api_key()
            time.sleep(delay * (i + 1))
    return None

if os.path.exists(labeled_comment_path):
    labeled_df = pd.read_csv(labeled_comment_path)
    if "Label" not in labeled_df.columns:
        labeled_df["Label"] = ""
else:
    labeled_df = df.copy()
    labeled_df["Label"] = ""

while True:
    unlabeled_indexes = labeled_df[labeled_df["Label"].isna() | (labeled_df["Label"] == "")].index.tolist()
    if not unlabeled_indexes:
        break

    for index in unlabeled_indexes:
        comment = labeled_df.loc[index, "Comment Text"]
        response = send_request_with_retry(comment)

        if response:
            labeled_df.loc[index, "Label"] = response
            print(f"Bình luận {index + 1}/{len(comments)}: {response}")
            labeled_df.to_csv(labeled_comment_path, index=False)
        else:
            print(f"Lỗi khi xử lý bình luận {index + 1}, sẽ thử lại sau...")
            time.sleep(5)

        time.sleep(1)

print("Hoàn thành! Tất cả bình luận đã có nhãn.")
