# encoding=utf-8
import argparse
import os
import re
import sys

import pymorphy2

_MORPH = pymorphy2.MorphAnalyzer()


def get_pos(token):
    return {parse.tag.POS for parse in _MORPH.parse(token)}


def regularize(text):
    text = re.sub("Ё", "Е", text)
    text = re.sub("ё", "е", text)
    text = re.sub(" +", " ", text)

    return text

def generic_score(true_dir, set_dir, output_stream):
    filenames = os.listdir(f"{true_dir}/generic/norm")

    n = 0
    hits = 0

    for name in filenames:
        gt = open(f"{true_dir}/generic/norm/{name}", encoding='utf-8').read().strip()

        gt = regularize(gt)

        gt_lines = gt.split('\n')

        if not os.path.exists(f"{set_dir}/{name}"):
            print(f'{name} doesn\'t exist')
            return 0.0

        sub = open(f"{set_dir}/{name}", encoding='utf-8').read().strip()

        sub = regularize(sub)

        sub_lines = sub.split('\n')

        for i, (gt, sub) in enumerate(zip(gt_lines, sub_lines)):
            n += 1

            if len(sub.split()) == 1 and len(get_pos(sub) & {'GRND', 'PRTF'}) > 0:
                print(gt, sub)

            if re.sub(" ", "", gt).lower() == re.sub(" ", "", sub).lower():
                hits += 1
            else:
                output_stream.write(f'[{name}/{i}] {gt}\t{sub}\n')

    return hits / n

def named_score(true_dir, set_dir, output_stream):
    filenames = os.listdir(f"{true_dir}/named/norm")

    n = 0
    hits = 0

    for name in filenames:
        gt = open(f"{true_dir}/named/norm/{name}", encoding='utf-8').read().strip()

        gt = regularize(gt)

        gt_lines = gt.split('\n')

        if not os.path.exists(f"{set_dir}/{name}"):
            print(f'{name} doesn\'t exist')
            return 0.0

        sub = open(f"{set_dir}/{name}", encoding='utf-8').read().strip()

        sub = regularize(sub)

        sub_lines = sub.split('\n')

        for i, (gt, sub) in enumerate(zip(gt_lines, sub_lines)):
            n += 1
            if gt == sub:
                hits += 1
            else:
                output_stream.write(f'[{name}/{i}] {gt}\t{sub}\n')

    return hits / n


if __name__ == "__main__":
    output_stream = sys.stdout

    parser = argparse.ArgumentParser()
    parser.add_argument('--true-dir')
    parser.add_argument('--prediction-dir')
    args = parser.parse_args()

    true_dir = args.true_dir
    predict_dir = args.prediction_dir
    output_stream = open(os.path.join(predict_dir, "scores.txt"), "w")

    set_1_dir = os.path.join(predict_dir, "generic")
    set_2_dir = os.path.join(predict_dir, "named")

    set_1_score = 0.0
    set_2_score = 0.0

    # Generic spans
    if not os.path.exists(set_1_dir) or len(os.listdir(set_1_dir)) == 0:
        set_1_score = 0.0
    else:
        set_1_score = generic_score(true_dir, set_1_dir, output_stream)

    # Named entities
    if not os.path.exists(set_2_dir) or len(os.listdir(set_2_dir)) == 0:
        set_2_score = 0.0
    else:
        set_2_score = named_score(true_dir, set_2_dir, output_stream)

    output_stream.write("set_1_score: %0.12f\n" % set_1_score)
    output_stream.write("set_2_score: %0.12f\n" % set_2_score)

    output_stream.close()
