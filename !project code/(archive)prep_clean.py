import os
import stanza
from stanza.utils.conll import CoNLL

# 1) Absolute path to your conllu
file_path = r"C:\ADI 25 winter\mlp\final project\mytest.conllu"

print("Working dir:", os.getcwd())
print("Checking file path:", file_path)
print("File exists?", os.path.exists(file_path))

# 2) Inspect first line’s raw bytes
with open(file_path, "rb") as f:
    first_line = f.readline()
print("First line bytes:", first_line)

# 3) If that looks correct, proceed
nlp = stanza.Pipeline("en", processors="tokenize,pos,lemma,depparse")

# 4) Load using CoNLL
conll_data = CoNLL.load_conll(file_path)
print(f"Loaded {len(conll_data)} sentences from {file_path}")

# 5) Convert each sentence to a raw string
raw_texts = [" ".join(token[1] for token in sent) for sent in conll_data]
predicted_docs = [nlp(text) for text in raw_texts]

# 6) Save predictions
out_path = "your_predictions.conllu"
CoNLL.write_doc2conll([doc.to_dict() for doc in predicted_docs], out_path)
print(f"✅ Predictions saved to {out_path}")
