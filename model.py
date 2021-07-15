import tensorflow as tf

# important constants

TOK_END_LEVEL = 0
TOK_END_BLOCK = 1
TOK_UNKNOWN = 2
TOK_PADDING = 3

LEVEL_DELIM = "---"
BLOCK_DELIM = "\n"
RESERVED_TOKENS = {
    TOK_END_LEVEL: LEVEL_DELIM,
    TOK_END_BLOCK: BLOCK_DELIM,
    TOK_UNKNOWN: "<unk>",
    TOK_PADDING: "<pad>",
}

VOCAB_SIZE = 512 + len(RESERVED_TOKENS)


def causal_attention_mask(n_src, n_dst):
    i = tf.range(n_dst)[:, None]
    j = tf.range(n_src)
    mask = i >= j - n_src + n_dst
    mask = mask[None, ...]
    return tf.cast(mask, tf.bool)


def positional_encoding_matrix(seq_len, depth, min_freq=1e-4):
    indices = tf.range(depth)
    sin_mask = tf.cast(indices % 2, tf.float32)
    cos_mask = 1 - sin_mask
    exponent = 2 * (indices // 2)
    exponent = tf.cast(exponent, tf.float32) / tf.cast(depth, tf.float32)
    freqs = min_freq ** exponent
    omega = tf.einsum("i,j->ij", tf.range(seq_len, dtype=tf.float32), freqs)
    return tf.math.cos(omega) * cos_mask + tf.math.sin(omega) * sin_mask


def embedding(tok_seq, embed_dim, vocab_size=VOCAB_SIZE):
    seq_len = tok_seq.shape[-1]
    tok_emb = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        name="lexical_embedding",
    )(tok_seq)
    pos_emb = positional_encoding_matrix(seq_len, embed_dim)
    pos_emb = pos_emb[None, ...]
    tok_emb = tf.keras.layers.LayerNormalization()(tok_emb)
    pos_emb = tf.keras.layers.LayerNormalization()(pos_emb)
    return tok_emb + pos_emb


def decoder_block(
    x, depth=64, ffdim=64, ffact=tf.nn.silu, attention_heads=2, dropout=0.1
):
    seq_len = x.shape[-2]

    def att(x):
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.MultiHeadAttention(attention_heads, x.shape[-1])(
            x, x, attention_mask=causal_attention_mask(seq_len, seq_len)
        )
        return tf.keras.layers.Dropout(dropout)(x)

    def ffn(x):
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dense(ffdim)(x)
        x = tf.keras.layers.Activation(ffact)(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.LayerNormalization()(x)
        return tf.keras.layers.Dense(depth)(x)

    return ffn(x + att(x))


def padder(seq_len, from_back=True):
    @tf.function
    def pad(tokens):
        excess = tf.shape(tokens)[-1] - seq_len
        if excess > 0:
            tokens = tokens[:seq_len]
        elif excess < 0:
            padding = [0, -excess] if from_back else [-excess, 0]
            tokens = tf.pad(
                tokens,
                paddings=[padding],
                mode="constant",
                constant_values=TOK_PADDING,
            )
        return tokens

    return pad


def tokenizer(vocab_size=VOCAB_SIZE, seq_len=None, from_back=True):
    @tf.function
    def tok_id(t):
        k = vocab_size - len(RESERVED_TOKENS)
        if t == "":
            k += TOK_END_BLOCK
        elif t == LEVEL_DELIM:
            k += TOK_END_LEVEL
        else:
            k = tf.strings.to_number(t, out_type=tf.int32)
            if not (0 <= k and k < vocab_size):
                k = vocab_size + TOK_UNKNOWN
        return k

    pad = padder(seq_len, from_back) if seq_len is not None else lambda x: x

    @tf.function
    def tokenize(s):
        tokens = tf.vectorized_map(tok_id, tf.strings.split(s))
        return pad(tokens)

    return tokenize


def transformer(seq_len, n_blocks, embed_dim, **kwargs):
    tok_seq = tf.keras.layers.Input(shape=[seq_len], dtype=tf.int32)
    outputs = embedding(tok_seq, embed_dim)
    for _ in range(n_blocks):
        outputs = decoder_block(outputs, **kwargs)
    return tf.keras.Model(tok_seq, outputs)


def sseq_loss(alpha=0.50, gamma=2.00, from_logits=True):
    @tf.function
    def loss(y, p, alpha=alpha, gamma=gamma, from_logits=from_logits):
        y = tf.one_hot(y, VOCAB_SIZE)
        alpha_factor = 1.0
        gamma_factor = 1.0
        inner_cost = tf.keras.losses.categorical_crossentropy(
            y, p, from_logits=from_logits
        )
        if from_logits:
            p = tf.nn.softmax(p)
        p = tf.cast(p, y.dtype)
        p_t = (y * p) + ((1 - y) * (1 - p))

        if alpha:
            alpha = tf.cast(alpha, y.dtype)
            alpha_factor = y * alpha + (1 - y) * (1 - alpha)
        if gamma:
            gamma = tf.cast(gamma, y.dtype)
            gamma_factor = tf.pow(1.0 - p_t, gamma)

        return tf.reduce_sum(inner_cost[..., None] * alpha_factor * gamma_factor)

    return loss


def generator(**kwargs):
    model = transformer(**kwargs)
    # define a synthetic objective: skip-thoughts
    embedding = model.layers[-1].output
    sseq = tf.keras.layers.Dense(VOCAB_SIZE, name="sseq")(embedding)
    model = tf.keras.Model(
        inputs=model.inputs, outputs=dict(sseq=sseq, embedding=embedding)
    )

    model.compile(
        loss=dict(
            sseq=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            # sseq=sseq_loss(alpha=0.50, gamma=2.00),
            embedding=None,
        ),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=dict(
            sseq=[
                tf.metrics.SparseTopKCategoricalAccuracy(k=1, name="acc@1"),
                tf.metrics.SparseTopKCategoricalAccuracy(k=4, name="acc@4"),
                tf.metrics.SparseTopKCategoricalAccuracy(k=8, name="acc@8"),
            ],
            embedding=None,
        ),
    )
    return model


def attach_stem_tokenizer(model, vocab_size=VOCAB_SIZE):
    inputs = tf.keras.layers.Input(shape=(), dtype=tf.string)
    seq_len = model.inputs[0].shape[-1]
    tokenize = tokenizer(seq_len, vocab_size)
    tok_seq = tf.keras.layers.Lambda(tokenize)(inputs)
    outputs = model(tok_seq)
    return tf.keras.Model(inputs, outputs)
