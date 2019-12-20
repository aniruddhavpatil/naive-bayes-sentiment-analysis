#!/usr/local/bin/python3
import sys
import os
import numpy as np
import operator
import re

def clean_text(text):
    cleaned_text = []
    text = text.lower()
    text = re.sub('<.*?>', '', text)
    text = text.split()
    for i in text:
        if 2 < len(i) < 20:
            cleaned_text.append(i.strip("\'?!():;\""))
    return cleaned_text

def read_data(data_dir, test=False):
    curr_dir = os.getcwd()
    data_dir_path = os.path.join(curr_dir, data_dir)
    data = []
    if test == False:
        spam_dir_path = os.path.join(data_dir_path, 'spam')
        spam_file_list = os.listdir(spam_dir_path)

        for file_name in spam_file_list:
            file_path = os.path.join(spam_dir_path, file_name)
            with open(file_path, errors='ignore') as f:
                text = f.read()
                label = 1
                data.append([file_name, clean_text(text), label])

        notspam_dir_path = os.path.join(data_dir_path, 'notspam')
        notspam_file_list = os.listdir(notspam_dir_path)
        for file_name in notspam_file_list:
            file_path = os.path.join(notspam_dir_path, file_name)
            with open(file_path, errors='ignore') as f:
                text = f.read()
                label = 0
                data.append([file_name, clean_text(text), label])
    else:
        file_list = os.listdir(data_dir_path)
        for file_name in file_list:
            file_path = os.path.join(data_dir_path, file_name)
            with open(file_path, errors='ignore') as f:
                text = f.read()
                data.append([file_name, clean_text(text)])
    return data


def build_vocabulary(data):
    V = [{}, {}]
    for datapoint in data:
        file_path = datapoint[0]
        words = datapoint[1]
        label = datapoint[2]
        for word in words:
            if word not in V[label]:
                V[label][word] = 1

            else:
                V[label][word] += 1
    return V

def test(data, V):
    positive_vocab_count = len(V[1])
    negative_vocab_count = len(V[0])

    num_positive_count = sum(V[1].values())
    num_negative_count = sum(V[0].values())

    m = 0.1
    prior_positive = 0.5
    prior_negative = 1 - prior_positive

    pos_denominator = num_positive_count + m * (positive_vocab_count + negative_vocab_count)
    neg_denominator = num_negative_count + m * (positive_vocab_count + negative_vocab_count)


    predictions = []
    for datapoint in data:
        prob_positive = np.log(prior_positive)
        prob_negative = np.log(prior_negative)
        file_name = datapoint[0]
        words = datapoint[1]
        for word in words:
            if word in V[1]:
                prob_positive += np.log((V[1][word] + m) / pos_denominator)
            else:
                if m > 0:
                    prob_positive += np.log(m / pos_denominator)
                else:
                    prob_positive -= np.log(pos_denominator)

            if word in V[0]:
                prob_negative += np.log((V[0][word] + m) / neg_denominator)
            else:
                if m > 0:
                    prob_negative += np.log(m / neg_denominator)
                else:
                    prob_negative -= np.log(neg_denominator)

        if prob_positive > prob_negative:
            predictions.append([file_name, 'spam'])
        else:
            predictions.append([file_name, 'notspam'])
    return predictions

def evaluate(gt_file, output_file):
    n_correct = 0
    gt = []
    pred = []
    pred_dict = {}
    with open(gt_file, 'r') as f:
        gt = f.readlines()
    with open(output_file, 'r') as f:
        pred = f.readlines()
    for line in pred:
        filename,label = line.split()
        pred_dict[filename] = label
    for line in gt:
        filename, label = line.split()
        if pred_dict[filename] == label:
            n_correct+=1
    print("Number correct:",n_correct)
    print("Total:",len(gt))
    print("Accuracy:", 100*n_correct/len(gt))

if __name__== "__main__":
    if (len(sys.argv) != 4):
        raise Exception("usage: ./spam.py training-directory testing-directory output-file")
    train_dir = sys.argv[1]
    test_dir = sys.argv[2]
    output_file = sys.argv[3]

    train_data = read_data(train_dir)
    test_data = read_data(test_dir, test=True)
    V = build_vocabulary(train_data)
    predictions = test(test_data, V)
    with open(output_file, 'w+') as f:
        for i in predictions:
            f.write(i[0] + ' ' + i[1] + '\n')

    evaluate('test-groundtruth.txt', output_file)
