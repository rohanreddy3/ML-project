import pandas as pd
import os

QUESTIONS = "data/raw/Questions3.csv"
TAGS = "data/raw/Tags3.csv"
ANSWERS = "data/raw/Answers3.csv"
OUT = "data/raw/dataset3.csv"

print("🔹 Loading files...")
q = pd.read_csv(QUESTIONS, encoding='latin1')
t = pd.read_csv(TAGS, encoding='latin1')
a = pd.read_csv(ANSWERS, encoding='latin1')

print("🔹 Merging datasets...")
if 'Id' in q.columns and 'Id' in t.columns:
    df = pd.merge(q, t, on='Id', how='inner')
else:
    df = pd.concat([q, t], axis=1)

if 'Title' in df.columns and 'Body' in df.columns:
    df['text'] = df['Title'].fillna('') + ' ' + df['Body'].fillna('')
elif 'Body' in df.columns:
    df['text'] = df['Body'].fillna('')
else:
    df['text'] = df.iloc[:, 1].astype(str) 

if 'Tag' in df.columns:
    df = df[['text', 'Tag']]
else:
    df = df.iloc[:, :2]
    df.columns = ['text', 'tags']

os.makedirs(os.path.dirname(OUT), exist_ok=True)
df.to_csv(OUT, index=False)
print(f"✅ Saved combined dataset to {OUT}")
print("Rows:", len(df))
print("Columns:", df.columns.tolist())
print("Sample data:")
print(df.head(3))
