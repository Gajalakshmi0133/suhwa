"""
Prepare dataset: convert per-frame CSV and saved .npy sequences into training arrays.
Writes X.npy (N, T, D) and y.npy (N,) where T is sequence length.
Usage:
  python tools/prepare_dataset.py --window 30 --stride 15 --out dataset/prepared
"""
import os
import argparse
import numpy as np
import glob


def load_sequences(seq_dir):
    files = glob.glob(os.path.join(seq_dir, '*.npy'))
    X = []
    y = []
    for f in files:
        arr = np.load(f)
        # label is prefix before first underscore
        base = os.path.basename(f)
        label = base.split('_')[0]
        X.append(arr)
        y.append(label)
    return X, y


def save_prepared(X, y, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # pad/crop to same length T
    lengths = [a.shape[0] for a in X]
    T = max(lengths)
    D = X[0].shape[1] if X else 63
    Xpad = np.zeros((len(X), T, D), dtype=np.float32)
    for i, a in enumerate(X):
        L = a.shape[0]
        if L <= T:
            Xpad[i, :L] = a
        else:
            Xpad[i] = a[:T]
    # map labels to ints
    labels = sorted(list(set(y)))
    label2i = {l:i for i,l in enumerate(labels)}
    yi = np.array([label2i[l] for l in y], dtype=np.int32)
    np.save(os.path.join(out_dir, 'X.npy'), Xpad)
    np.save(os.path.join(out_dir, 'y.npy'), yi)
    with open(os.path.join(out_dir, 'labels.txt'), 'w', encoding='utf8') as fh:
        for l in labels:
            fh.write(l + '\n')
    print('Saved prepared dataset to', out_dir)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seq-dir', default='dataset/sequences')
    p.add_argument('--out', default='dataset/prepared')
    args = p.parse_args()
    X, y = load_sequences(args.seq_dir)
    if not X:
        print('No sequences found in', args.seq_dir)
        exit(1)
    save_prepared(X, y, args.out)
