import pandas as pd

df1 = pd.read_csv('/home/lamonade/Engineering/KSH_semantic_analysis/DATA/support_comments_segmented_2.csv')
df2 = pd.read_csv('/home/lamonade/Engineering/KSH_semantic_analysis/DATA/labeled_comments_segmented.csv')
# df3 = pd.read_csv('/home/lamonade/Engineering/KSH_semantic_analysis/DATA/labeled_comments_segmented_3.csv')
# df4 = pd.read_csv('/home/lamonade/Engineering/KSH_semantic_analysis/DATA/labeled_comments_segmented_4.csv')

df_merged = pd.concat([df1, df2], ignore_index=True)
df_merged['ID'] = range(1, len(df_merged) + 1)

df_merged.to_csv('/home/lamonade/Engineering/KSH_semantic_analysis/DATA/labeled_comments_segmented.csv', index=False)
