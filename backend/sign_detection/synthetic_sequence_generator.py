import os
from pathlib import Path
import cv2


ALPHABET_DIR = Path(__file__).resolve().parents[2] / 'asl_alphabet_test'
OUT_DIR = Path(__file__).resolve().parents[2] / 'dataset' / 'synthetic_sequences'
SENTENCES_FILE = Path(__file__).resolve().parents[2] / 'dataset' / 'synthetic_sentences.txt'


def char_to_filename(ch: str):
    ch = ch.lower()
    if ch == ' ':
        return 'space_test.jpg'
    if ch == '' or ch == '\n':
        return None
    name = f"{ch.upper()}_test.jpg"
    return name


def make_sequence(sentence: str, out_path: Path, frames_per_letter=3, size=(64, 64)):
    out_path.mkdir(parents=True, exist_ok=True)
    frame_idx = 0
    for ch in sentence:
        fname = char_to_filename(ch)
        if not fname:
            continue
        src = ALPHABET_DIR / fname
        if not src.exists():
            print('Missing source image for', ch, src)
            continue
        img = cv2.imread(str(src))
        if img is None:
            print('Failed to read', src)
            continue
        img = cv2.resize(img, size)
        for _ in range(frames_per_letter):
            frame_idx += 1
            out_file = out_path / f"frame_{frame_idx:04d}.jpg"
            cv2.imwrite(str(out_file), img)
    with open(out_path / 'transcript.txt', 'w', encoding='utf8') as f:
        f.write(sentence.strip().lower())


def generate_all(sentences_file=SENTENCES_FILE, out_dir=OUT_DIR, frames_per_letter=3):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(sentences_file, 'r', encoding='utf8') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    for i, s in enumerate(lines):
        seq_dir = out_dir / f'seq_{i:04d}'
        print('Generating', seq_dir, '->', s)
        make_sequence(s, seq_dir, frames_per_letter=frames_per_letter)


if __name__ == '__main__':
    print('Generating synthetic sequences from', SENTENCES_FILE)
    generate_all()
