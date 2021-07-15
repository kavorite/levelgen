import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

from model import LEVEL_DELIM, generator, tokenizer


def read_level_ngrams(level_path, seq_len):
    tokenize = tokenizer()

    def parse_ngrams(chunk):
        ctx = tokenize(tf.convert_to_tensor(chunk).numpy())
        while len(ctx) > seq_len:
            for i in range(len(ctx) - seq_len - 1):
                window = ctx[i : i + seq_len + 1]
                source = window[:-1]
                target = window[1:]
                yield source, target
            ctx = ctx[seq_len:]

    def token_ngrams():
        with tf.io.gfile.GFile(level_path) as istrm:
            chunk_size = 8192
            chunk = ""
            while line := istrm.readline():
                chunk += line
                if len(chunk) > chunk_size:
                    yield from parse_ngrams(chunk)
                    chunk = ""
            yield from parse_ngrams(chunk)

    return tf.data.Dataset.from_generator(
        token_ngrams,
        output_signature=(
            tf.TensorSpec(shape=[seq_len], dtype=tf.int32),
            tf.TensorSpec(shape=[seq_len], dtype=tf.int32),
        ),
    )


MODEL_CONFIG = dict(n_blocks=16, embed_dim=64, depth=64, seq_len=768)


if __name__ == "__main__":
    with tf.io.gfile.GFile("./levels.txt") as istrm:
        content = istrm.read()
        total_tokens = len(content.split())
        total_levels = content.count(LEVEL_DELIM) + 1

    seq_len = MODEL_CONFIG["seq_len"]
    batch_size = 16
    dataset = (
        read_level_ngrams("./levels.txt", MODEL_CONFIG["seq_len"])
        .batch(batch_size)
        .cache()
        .shuffle(total_levels)
        .repeat()
    )
    model = generator(**MODEL_CONFIG)
    model.fit(
        dataset,
        verbose=1,
        shuffle=False,
        steps_per_epoch=int(
            tf.math.ceil((total_tokens - seq_len) / total_levels / batch_size)
        ),
        epochs=2,
    )

    model.save_weights("./mario.ckpt")
