import argparse
import os
from collections import Counter
from sys import stderr, stdout

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

from model import BLOCK_DELIM, LEVEL_DELIM, RESERVED_TOKENS, VOCAB_SIZE
from train import MODEL_CONFIG, generator, tokenizer


def beam_search(probs, width):
    sequences = [[[], 0.0]]
    for row in probs:
        search_tree = []
        for seq, p in sequences:
            for j in range(len(row)):
                p_branch = tf.math.log(row[j])
                p_branch = p - p_branch
                branch = (*seq, j), p
                search_tree.append(branch)
        ordered = sorted(search_tree, key=lambda pair: pair[1])[::-1]
        sequences = ordered[:width]

    return sequences


if __name__ == "__main__":
    stderr.write("loading levels...\n")
    levels = []
    with tf.io.gfile.GFile("./levels.txt") as istrm:
        level = ""
        while line := istrm.readline():
            if LEVEL_DELIM in line:
                levels.append(level)
                level = ""
            else:
                level += line

    parser = argparse.ArgumentParser()
    parser.add_argument("--level-prompt", type=lambda t: levels[int(t)], default=-1)
    parser.add_argument("--output-length", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--normalize", action="store_false", default=True)
    parser.add_argument("--stochastic", action="store_false", default=True)
    parser.add_argument("--lookahead", type=int, default=1)
    parser.add_argument("--search-width", type=int, default=8)
    args = parser.parse_args()

    stderr.write("loading model...\n")
    model = generator(**MODEL_CONFIG)
    model.load_weights("./mario.ckpt")
    model = tf.keras.Model(model.inputs, model.output["sseq"])

    maxlen = args.output_length or model.output.shape[-2]
    maxlen = min(maxlen, model.output.shape[-2])

    stderr.write("generating topology...\n")
    prompt = levels[args.level_prompt]
    prompt = tokenizer(seq_len=model.input.shape[-1])(prompt)
    prompt = list(prompt.numpy())

    if args.normalize:
        tok_hist = Counter(prompt)
        tok_hist = tf.constant([tok_hist[i] for i in range(VOCAB_SIZE)])
        tok_hist = tf.cast((tok_hist + 1) / tf.reduce_max(tok_hist + 1), tf.float32)
    else:
        tok_hist = tf.ones(VOCAB_SIZE, tf.float32)

    output = []
    while len(output) < maxlen:
        yhats = tf.squeeze(model.predict(tf.expand_dims(prompt, 0)))
        yhats = yhats[: args.lookahead, ...]
        yhats = yhats / tok_hist[None, ...]
        yhats = tf.nn.softmax(yhats / args.temperature, axis=-1)
        candidates = beam_search(probs=yhats, width=args.search_width)
        branches, scores = zip(*candidates)
        if args.stochastic:
            selection = tf.random.categorical(logits=scores[None, :], num_samples=1)
            selection = tf.squeeze(selection)
        else:
            selection = tf.argmax(scores)
        selection = branches[selection]
        output.extend(selection)
        prompt.extend(selection)
        if (excess := len(prompt) - model.input.shape[-1]) > 0:
            prompt = prompt[excess:]

    def detokenize(i):
        if (k := VOCAB_SIZE - i) in RESERVED_TOKENS:
            t = RESERVED_TOKENS[k]
            if t == LEVEL_DELIM:
                t = f"{BLOCK_DELIM}{t}{BLOCK_DELIM}"
            return t
        else:
            return str(i)

    stdout.write(" ".join(map(detokenize, output)))
