import argparse
import os
import random
random.seed(1234)

def train_test_split(path):
    l = [sub for sub in os.listdir(path) if os.path.isdir(os.path.join(path, sub))]
    num_total = len(l)
    random.shuffle(l)
    num_train = int(0.8 * num_total)
    train_char = l[:num_train]
    test_char = l[num_train:]
    print(f"Total number of char: {num_total}, train: {num_train}, test: {num_total - num_train}")
    return train_char, test_char


def write_list(char_list, path):
    with open(path, 'w') as f:
        f.writelines(f"{char}\n" for char in char_list)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="data/chinese")
    args = parser.parse_args()

    train_char, test_char = train_test_split(args.path)
    write_list(train_char, os.path.join(args.path, "train_list.txt"))
    write_list(test_char, os.path.join(args.path, "test_list.txt"))