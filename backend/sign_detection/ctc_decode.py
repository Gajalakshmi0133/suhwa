import numpy as np
from pathlib import Path

# Local alphabet mapping (must match training mapping in train_sequence_model)
ALPHABET = list('abcdefghijklmnopqrstuvwxyz') + [' ']
NUM_CLASSES = len(ALPHABET) + 1


def indices_to_text(indices):
    # indices is a 1D array of ints (may contain -1 for padding)
    chars = []
    for i in indices:
        if i < 0:
            continue
        if i >= len(ALPHABET):
            # blank or out-of-range
            continue
        chars.append(ALPHABET[i])
    return ''.join(chars)


def greedy_decode(y_pred):
    # y_pred: (batch, time, num_classes)
    import tensorflow as tf
    from keras import backend as K
    input_length = np.ones((y_pred.shape[0],), dtype=np.int32) * y_pred.shape[1]
    decoded, log_prob = K.ctc_decode(y_pred, input_length, greedy=True)
    decoded_dense = tf.sparse.to_dense(decoded[0], default_value=-1).numpy()
    texts = [indices_to_text(row) for row in decoded_dense]
    return texts


def beam_search_decode(y_pred, beam_width=10, top_paths=1):
    # y_pred: (batch, time, num_classes)
    import tensorflow as tf
    from keras import backend as K
    input_length = np.ones((y_pred.shape[0],), dtype=np.int32) * y_pred.shape[1]
    decoded, log_prob = K.ctc_decode(y_pred, input_length, greedy=False, beam_width=beam_width, top_paths=top_paths)
    # decoded is list of top_paths sparse tensors
    results = []
    for p in range(top_paths):
        try:
            decoded_dense = tf.sparse.to_dense(decoded[p], default_value=-1).numpy()
        except (TypeError, ValueError):
            # sometimes ctc_decode returns a dense tensor directly
            decoded_dense = decoded[p].numpy()
        results.append([indices_to_text(row) for row in decoded_dense])
    # return top_paths lists; common case top_paths=1 -> return first list
    if top_paths == 1:
        return results[0]
    return results


def beam_search_with_lm(y_pred, lm, beam_width=10, top_paths=5, lm_weight=1.0):
    """Run K.ctc_decode with multiple paths, then rescore with a language model.
    Returns best-scoring string per batch element.
    """
    import tensorflow as tf
    from keras import backend as K
    input_length = np.ones((y_pred.shape[0],), dtype=np.int32) * y_pred.shape[1]
    decoded, log_prob = K.ctc_decode(y_pred, input_length, greedy=False, beam_width=beam_width, top_paths=top_paths)
    best_results = []
    for batch_idx in range(y_pred.shape[0]):
        candidates = []
        for p in range(top_paths):
            try:
                decoded_dense = tf.sparse.to_dense(decoded[p], default_value=-1).numpy()
            except (TypeError, ValueError):
                decoded_dense = decoded[p].numpy()
            seq = decoded_dense[batch_idx]
            text = indices_to_text(seq)
            lm_score = lm.score(text) if lm is not None else 0.0
            # combine model log-prob (if available) with LM score
            model_lp = float(log_prob[p][batch_idx]) if (log_prob is not None and hasattr(log_prob, '__len__')) else 0.0
            total_score = model_lp + lm_weight * lm_score
            candidates.append((total_score, text))
        # pick best
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_results.append(candidates[0][1] if candidates else '')
    return best_results


def decode_from_model(model, X_batch, method='beam', beam_width=10):
    # X_batch: (batch, time, h, w, c)
    y_pred = model.predict(X_batch)
    if method == 'greedy':
        return greedy_decode(y_pred)
    return beam_search_decode(y_pred, beam_width=beam_width)


if __name__ == '__main__':
    print('This module provides CTC decode helpers.')
