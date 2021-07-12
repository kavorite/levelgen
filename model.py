import tensorflow as tf


def causal_attention_mask(batch_size, n_dst, n_src, dtype=tf.bool):
    i = tf.range(n_dst)[:, None]
    j = tf.range(n_src)
    m = i >= j - n_src + n_dst
    mask = tf.cast(m, dtype)
    mask = tf.reshape(mask, [1, n_dst, n_src])
    mult = tf.concat(
        [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
    )
    return tf.tile(mask, mult)


def embedding(tok_seq, vocab_size, embed_dim):
    tok_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
    pos_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)

    @tf.function
    def omega(k):
        return 1 / (10_000 ** (2 * k / embed_dim))

    poses = tf.range(start=0, limit=tf.shape(tok_seq)[-1], delta=1)
    rates = tf.map_fn(omega, poses)
    sines = tf.math.sin(rates)
    coses = tf.math.cos(rates)
    pos_enc = tf.where(poses % 2 == 0, sines, coses)
    return pos_emb(pos_enc) + tok_emb(tok_seq)


def decoder_block(x, model_dim, hidden_dim=64, num_heads=2, dropout=0.1):
    batch_size, seq_len = tf.shape(x)
    att_mask = causal_attention_mask(batch_size, seq_len, seq_len)

    def att(x):
        x = tf.keras.layers.MultiHeadAttention(
            num_heads, model_dim, attention_mask=att_mask
        )(x, x)
        return tf.keras.layers.Dropout(dropout)(x)

    def ffn(x):
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dense(hidden_dim)(x)
        x = tf.keras.layers.Activation(tf.nn.silu)(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Dense(model_dim)(x)

    return ffn(x + att(x))


TOK_END_LEVEL = -1
TOK_END_BLOCK = -2
TOK_UNKNOWN = -3
TOK_PADDING = 37


def tokenizer(seq_len):
    def tokenize(s):
        def tok_id(t):
            if tf.strings.regex_full_match("[0-9]+"):
                return tf.strings.to_number(t, out_type=tf.int16)
            elif tf.strings.regex_full_match("(\r?\n){1}"):
                return TOK_END_BLOCK
            elif tf.strings.regex_full_match("---\r?\n"):
                return TOK_END_LEVEL
            else:
                return TOK_UNKNOWN

        ragged = tf.map_fn(tok_id, tf.strings.split(s))
        padded = ragged.to_tensor(default_value=TOK_PADDING, shape=[None, seq_len])
        return padded

    return tokenize


def transformer(seq_len, n_blocks, **kwargs):
    inputs = tf.keras.layers.Input(shape=(), dtype=tf.string)
    tok_seq = tokenizer(seq_len)(inputs)
    outputs = embedding(tok_seq)
    for _ in range(n_blocks):
        outputs = decoder_block(outputs, **kwargs)
    return tf.keras.Model(inputs, outputs)
