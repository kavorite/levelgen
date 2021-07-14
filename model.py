import tensorflow as tf

# important constants

# keyspace needs to be sized for encoding special, negative tokens as well as tile indices
VOCAB_SIZE = 512 + 3
TOK_END_LEVEL = -1
TOK_END_BLOCK = -2
TOK_UNKNOWN = -3
TOK_PADDING = 37  # SMW "air" tile ID
LEVEL_DELIM = "---"


def causal_attention_mask(n_dst, n_src, dtype=tf.bool):
    i = tf.range(n_dst)[:, None]
    j = tf.range(n_src)
    m = i >= j - n_src + n_dst
    mask = tf.cast(m, dtype)
    return tf.reshape(mask, [1, n_dst, n_src])


def positional_encoding_matrix(seq_len, depth, min_freq=1e-4):
    mask = tf.range(depth)
    sin_mask = tf.cast(mask % 2, tf.float32)
    cos_mask = 1 - sin_mask
    exponent = 2 * (mask // 2)
    exponent = tf.cast(exponent, tf.float32) / tf.cast(depth, tf.float32)
    freqs = min_freq ** exponent
    omega = tf.einsum("i,j->ij", tf.range(seq_len, dtype=tf.float32), freqs)
    return tf.math.cos(omega) * cos_mask + tf.math.sin(omega) * sin_mask


def embedding(tok_seq, embed_dim, vocab_size=VOCAB_SIZE):
    seq_len = tok_seq.shape[-1]
    tok_emb = tf.keras.layers.Embedding(
        input_dim=vocab_size, output_dim=embed_dim, name="lexical_embedding"
    )(tok_seq)
    return tok_emb + positional_encoding_matrix(seq_len, embed_dim)[None, ...]


def decoder_block(x, depth=64, ff_dim=64, attention_heads=2, dropout=0.1):
    seq_len = x.shape[-2]
    attention_mask = causal_attention_mask(seq_len, seq_len)

    def att(x):
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.MultiHeadAttention(attention_heads, x.shape[-1])(
            x, x, attention_mask=attention_mask[None, ...]
        )
        return tf.keras.layers.Dropout(dropout)(x)

    def ffn(x):
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dense(ff_dim)(x)
        x = tf.keras.layers.Activation(tf.nn.silu)(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.LayerNormalization()(x)
        return tf.keras.layers.Dense(depth)(x)

    return ffn(x + att(x))


def padder(seq_len):
    @tf.function
    def pad(tokens):
        excess = tf.shape(tokens)[-1] - seq_len
        if excess > 0:
            tokens = tokens[:seq_len]
        elif excess < 0:
            tokens = tf.pad(
                tokens,
                paddings=[[0, -excess]],
                mode="constant",
                constant_values=TOK_PADDING,
            )
        return tokens

    return pad


def tokenizer(seq_len=None, vocab_size=VOCAB_SIZE):
    @tf.function
    def tokenize(s):
        pad = padder(seq_len) if seq_len is not None else lambda x: x

        @tf.function
        def tok_id(t):
            if tf.strings.regex_full_match(t, "[0-9]+"):
                k = tf.strings.to_number(t, out_type=tf.int32)
                if not (0 <= k and k < vocab_size):
                    return vocab_size + TOK_UNKNOWN
                else:
                    return k
            elif tf.math.equal(t, ""):
                return vocab_size + TOK_END_BLOCK
            elif tf.math.equal(t, LEVEL_DELIM):
                return vocab_size + TOK_END_LEVEL
            else:
                return vocab_size + TOK_UNKNOWN

        tokens = tf.map_fn(tok_id, tf.strings.split(s), fn_output_signature=tf.int32)
        return pad(tokens)

    return tokenize


def transformer(seq_len, n_blocks, embed_dim, **kwargs):
    tok_seq = tf.keras.layers.Input(shape=[seq_len], dtype=tf.int32)
    outputs = embedding(tok_seq, embed_dim)
    for _ in range(n_blocks):
        outputs = decoder_block(outputs, **kwargs)
    outputs = tf.keras.layers.LayerNormalization()(outputs)
    return tf.keras.Model(tok_seq, outputs)


def with_stem_tokenizer(model, vocab_size=VOCAB_SIZE):
    inputs = tf.keras.layers.Input(shape=(), dtype=tf.string)
    seq_len = model.inputs[0].shape[-1]
    tokenize = tokenizer(seq_len, vocab_size)
    tok_seq = tf.keras.layers.Lambda(tokenize)(inputs)
    outputs = model(tok_seq)
    return tf.keras.Model(inputs, outputs)
