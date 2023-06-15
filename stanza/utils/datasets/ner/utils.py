"""
Utils for the processing of NER datasets

These can be invoked from either the specific dataset scripts
or the entire prepare_ner_dataset.py script
"""

from collections import defaultdict
import json
import os
import random

import stanza.utils.datasets.ner.prepare_ner_file as prepare_ner_file

SHARDS = ('train', 'dev', 'test')

def convert_bio_to_json(base_input_path, base_output_path, short_name, suffix="bio", shard_names=SHARDS):
    """
    Convert BIO files to json

    It can often be convenient to put the intermediate BIO files in
    the same directory as the output files, in which case you can pass
    in same path for both base_input_path and base_output_path.
    """
    for input_shard, output_shard in zip(shard_names, SHARDS):
        input_filename = os.path.join(base_input_path, '%s.%s.%s' % (short_name, input_shard, suffix))
        if not os.path.exists(input_filename):
            alt_filename = os.path.join(base_input_path, '%s.%s' % (input_shard, suffix))
            if os.path.exists(alt_filename):
                input_filename = alt_filename
            else:
                raise FileNotFoundError('Cannot find %s component of %s in %s or %s' % (output_shard, short_name, input_filename, alt_filename))
        output_filename = os.path.join(base_output_path, '%s.%s.json' % (short_name, output_shard))
        print("Converting %s to %s" % (input_filename, output_filename))
        prepare_ner_file.process_dataset(input_filename, output_filename)

def get_tags(datasets):
    """
    return the set of tags used in these datasets

    datasets is expected to be train, dev, test but could be any list
    """
    tags = set()
    for dataset in datasets:
        for sentence in dataset:
            for word, tag in sentence:
                tags.add(tag)
    return tags

def write_sentences(output_filename, dataset):
    """
    Write exactly one output file worth of dataset
    """
    os.makedirs(os.path.split(output_filename)[0], exist_ok=True)
    with open(output_filename, "w", encoding="utf-8") as fout:
        for sentence in dataset:
            for word in sentence:
                fout.write("%s\t%s\n" % word)
            fout.write("\n")

def write_dataset(datasets, output_dir, short_name, suffix="bio"):
    """
    write all three pieces of a dataset to output_dir

    datasets should be 3 lists: train, dev, test
    each list should be a list of sentences
    each sentence is a list of pairs: word, tag

    after writing to .bio files, the files will be converted to .json
    """
    for shard, dataset in zip(SHARDS, datasets):
        output_filename = os.path.join(output_dir, "%s.%s.%s" % (short_name, shard, suffix))
        write_sentences(output_filename, dataset)

    convert_bio_to_json(output_dir, output_dir, short_name, suffix)


def read_tsv(filename, text_column, annotation_column, remap_fn=None, skip_comments=True, keep_broken_tags=False, keep_all_columns=False):
    """
    Read sentences from a TSV file

    Returns a list of list of (word, tag)

    If keep_broken_tags==True, then None is returned for a missing.  Otherwise, an IndexError is thrown
    """
    with open(filename, encoding="utf-8") as fin:
        lines = fin.readlines()

    lines = [x.strip() for x in lines]

    sentences = []
    current_sentence = []
    for line_idx, line in enumerate(lines):
        if not line:
            if current_sentence:
                sentences.append(current_sentence)
                current_sentence = []
            continue
        if skip_comments and line.startswith("#"):
            continue

        pieces = line.split("\t")
        try:
            word = pieces[text_column]
        except IndexError as e:
            raise IndexError("Could not find word index %d at line %d" % (text_column, line_idx)) from e
        if word == '\x96':
            # this happens in GermEval2014 for some reason
            continue
        try:
            tag = pieces[annotation_column]
        except IndexError as e:
            if keep_broken_tags:
                tag = None
            else:
                raise IndexError("Could not find tag index %d at line %d" % (annotation_column, line_idx)) from e
        if remap_fn:
            tag = remap_fn(tag)

        if keep_all_columns:
            pieces[annotation_column] = tag
            current_sentence.append(pieces)
        else:
            current_sentence.append((word, tag))

    if current_sentence:
        sentences.append(current_sentence)

    return sentences

def random_shuffle_directory(input_dir, output_dir, short_name):
    input_files = os.listdir(input_dir)
    input_files = sorted(input_files)
    random_shuffle_files(input_dir, input_files, output_dir, short_name)

def random_shuffle_files(input_dir, input_files, output_dir, short_name):
    """
    Shuffle the files into different chunks based on their filename

    The first piece of the filename, split by ".", is used as a random seed.

    This will make it so that adding new files or using a different
    annotation scheme (assuming that's encoding in pieces of the
    filename) won't change the distibution of the files
    """
    input_keys = {}
    for f in input_files:
        seed = f.split(".")[0]
        if seed in input_keys:
            raise ValueError("Multiple files with the same prefix: %s and %s" % (input_keys[seed], f))
        input_keys[seed] = f
    assert len(input_keys) == len(input_files)

    train_files = []
    dev_files = []
    test_files = []

    for filename in input_files:
        seed = filename.split(".")[0]
        # "salt" the filenames when using as a seed
        # definitely not because of a dumb bug in the original implementation
        seed = seed + ".txt.4class.tsv"
        random.seed(seed, 2)
        location = random.random()
        if location < 0.7:
            train_files.append(filename)
        elif location < 0.8:
            dev_files.append(filename)
        else:
            test_files.append(filename)

    print("Train files: %d  Dev files: %d  Test files: %d" % (len(train_files), len(dev_files), len(test_files)))
    assert len(train_files) + len(dev_files) + len(test_files) == len(input_files)

    file_lists = [train_files, dev_files, test_files]
    datasets = []
    for files in file_lists:
        dataset = []
        for filename in files:
            dataset.extend(read_tsv(os.path.join(input_dir, filename), 0, 1))
        datasets.append(dataset)

    write_dataset(datasets, output_dir, short_name)

def random_shuffle_by_prefixes(input_dir, output_dir, short_name, prefix_map):
    input_files = os.listdir(input_dir)
    input_files = sorted(input_files)

    file_divisions = defaultdict(list)
    for filename in input_files:
        for division in prefix_map.keys():
            for prefix in prefix_map[division]:
                if filename.startswith(prefix):
                    break
            else: # for/else is intentional
                continue
            break
        else: # yes, stop asking
            raise ValueError("Could not assign %s to any of the divisions in the prefix_map" % filename)
        #print("Assigning %s to %s because of %s" % (filename, division, prefix))
        file_divisions[division].append(filename)

    for division in file_divisions.keys():
        print()
        print("Processing %d files from %s" % (len(file_divisions[division]), division))
        random_shuffle_files(input_dir, file_divisions[division], output_dir, "%s-%s" % (short_name, division))

    dataset_divisions = ["%s-%s" % (short_name, division) for division in file_divisions]
    combine_dataset(output_dir, output_dir, dataset_divisions, short_name)

def combine_dataset(input_dir, output_dir, input_datasets, output_dataset):
    datasets = []
    for shard in SHARDS:
        full_dataset = []
        for input_dataset in input_datasets:
            input_filename = "%s.%s.json" % (input_dataset, shard)
            input_path = os.path.join(input_dir, input_filename)
            with open(input_path, encoding="utf-8") as fin:
                dataset = json.load(fin)
                converted = [[(word['text'], word['ner']) for word in sentence] for sentence in dataset]
                full_dataset.extend(converted)
        datasets.append(full_dataset)
    write_dataset(datasets, output_dir, output_dataset)

def read_prefix_file(destination_file):
    """
    Read a prefix file such as the one for the Worldwide dataset

    the format should be

    africa:
    af_
    ...

    asia:
    cn_
    ...
    """
    destination = None
    known_prefixes = set()
    prefixes = []

    prefix_map = {}
    with open(destination_file, encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line.startswith("#"):
                continue
            if not line:
                continue
            if line.endswith(":"):
                if destination is not None:
                    prefix_map[destination] = prefixes
                prefixes = []
                destination = line[:-1].strip().lower().replace(" ", "_")
            else:
                if not destination:
                    raise RuntimeError("Found a prefix before the first label was assigned when reading %s" % destination_file)
                prefixes.append(line)
                if line in known_prefixes:
                    raise RuntimeError("Found the same prefix twice! %s" % line)
                known_prefixes.add(line)

        if destination and prefixes:
            prefix_map[destination] = prefixes

    return prefix_map
