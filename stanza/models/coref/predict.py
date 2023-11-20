import argparse

import json
import torch
from tqdm import tqdm

from stanza.models.coref.model import CorefModel
from stanza.models.coref.tokenizer_customization import *


def build_doc(doc: dict, model: CorefModel) -> dict:
    filter_func = TOKENIZER_FILTERS.get(model.config.bert_model,
                                        lambda _: True)
    token_map = TOKENIZER_MAPS.get(model.config.bert_model, {})

    word2subword = []
    subwords = []
    word_id = []
    for i, word in enumerate(doc["cased_words"]):
        tokenized_word = (token_map[word]
                          if word in token_map
                          else model.tokenizer.tokenize(word))
        tokenized_word = list(filter(filter_func, tokenized_word))
        word2subword.append((len(subwords), len(subwords) + len(tokenized_word)))
        subwords.extend(tokenized_word)
        word_id.extend([i] * len(tokenized_word))
    doc["word2subword"] = word2subword
    doc["subwords"] = subwords
    doc["word_id"] = word_id

    doc["head2span"] = []
    if "speaker" not in doc:
        doc["speaker"] = ["_" for _ in doc["cased_words"]]
    doc["word_clusters"] = []
    doc["span_clusters"] = []

    return doc


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("experiment")
    argparser.add_argument("input_file")
    argparser.add_argument("output_file")
    argparser.add_argument("--config-file", default="config.toml")
    argparser.add_argument("--batch-size", type=int,
                           help="Adjust to override the config value if you're"
                                " experiencing out-of-memory issues")
    argparser.add_argument("--weights",
                           help="Path to file with weights to load."
                                " If not supplied, in the latest"
                                " weights of the experiment will be loaded;"
                                " if there aren't any, an error is raised.")
    args = argparser.parse_args()

    model = CorefModel.load_model(path=args.weights,
                                  map_location="cpu",
                                  ignore={"bert_optimizer", "general_optimizer",
                                          "bert_scheduler", "general_scheduler"})
    if args.batch_size:
        model.config.a_scoring_batch_size = args.batch_size
    model.training = False

    try:
        with open(args.input_file, encoding="utf-8") as fin:
            input_data = json.load(fin)
    except json.decoder.JSONDecodeError:
        # read the old jsonlines format if necessary
        with open(args.input_file, encoding="utf-8") as fin:
            text = "[" + ",\n".join(fin) + "]"
        input_data = json.loads(text)
    docs = [build_doc(doc, model) for doc in input_data]

    with torch.no_grad():
        for doc in tqdm(docs, unit="docs"):
            result = model.run(doc)
            doc["span_clusters"] = result.span_clusters
            doc["word_clusters"] = result.word_clusters

            for key in ("word2subword", "subwords", "word_id", "head2span"):
                del doc[key]

    with open(args.output_file, mode="w") as fout:
        for doc in docs:
            json.dump(doc, fout)
