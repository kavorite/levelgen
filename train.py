import tensorflow as tf

from model import LEVEL_DELIM, VOCAB_SIZE, tokenizer, transformer

tf.config.run_functions_eagerly(True)


def read_level_seqs(level_path, seq_len):
    tokenize = tokenizer()

    def token_windows():
        with tf.io.gfile.GFile(level_path) as istrm:
            ctx = []
            while line := istrm.readline():
                ctx.extend(tokenize(tf.convert_to_tensor(line)).numpy())
                if len(ctx) > seq_len:
                    for i in range(len(ctx) - seq_len - 1):
                        window = ctx[i : i + seq_len + 1]
                        source = window[:-1]
                        target = window[1:]
                        yield source, target
                    ctx = ctx[seq_len + 1 :]

    return tf.data.Dataset.from_generator(
        token_windows,
        output_signature=(
            tf.TensorSpec(shape=[seq_len], dtype=tf.int32),
            tf.TensorSpec(shape=[seq_len], dtype=tf.int32),
        ),
    ).prefetch(tf.data.AUTOTUNE)


with tf.io.gfile.GFile("./levels.txt") as istrm:
    content = istrm.read()
    total_tokens = len(content.split())
    total_levels = content.count(LEVEL_DELIM)

seq_len = int(tf.math.ceil(total_tokens / total_levels))
batch_size = 32
dataset = read_level_seqs("./levels.txt", seq_len).batch(batch_size).repeat()
model = transformer(n_blocks=8, embed_dim=128, depth=128, seq_len=seq_len)
# define a synthetic objective: skip-thoughts
succ_seq = tf.keras.layers.Dense(VOCAB_SIZE)(model.layers[-1].output)
model = tf.keras.Model(inputs=model.inputs, outputs=[succ_seq, *model.outputs])
succ_seq_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(
    loss=[succ_seq_loss, None],
    optimizer=tf.keras.optimizers.Adam(),
    # metrics=[tf.metrics.AUC(name="auc", from_logits=True)],
)
model.fit(
    dataset,
    verbose=1,
    steps_per_epoch=int(
        tf.math.ceil((total_tokens - seq_len) / total_levels / batch_size)
    ),
)
