import pandas as pd
import py_vncorenlp
from tqdm import tqdm

df = pd.read_csv("DATA/labeled_comments3.csv")

# py_vncorenlp.download_model(save_dir='DATA_PROC/proc_engine/vncorenlp')

rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='/home/lamonade/Engineering/KSH_semantic_analysis/DATA_PROC/proc_engine/vncorenlp')

def segment_text(text: str):
    segment_list = rdrsegmenter.word_segment(text)
    return " ".join(segment_list)

tqdm.pandas()
df["Comment Text"] = df["Comment Text"].apply(segment_text)

df.to_csv("/home/lamonade/Engineering/KSH_semantic_analysis/DATA/labeled_comments_segmented_3.csv", index=False)
