import tensorflow as tf
import keras
from tensorflow.keras import layers

# Parameters
vocab_size = 10000       # Size of the vocabulary
max_len = 128            # Maximum sequence length
embed_dim = 256          # Embedding dimension
num_heads = 4            # Number of attention heads
head_size = 64           # Key dimension per head
ff_dim = 512             # Size of feed-forward network
num_layers = 4           # How many transformer blocks
dropout_rate = 0.1       # Dropout for regularization

# Causal Mask Function
def create_causal_attention_mask(batch_size, seq_len):
    mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)  # Lower triangular matrix
    mask = tf.reshape(mask, (1, 1, seq_len, seq_len))  # shape: (batch_size, num_heads, seq_len, seq_len)
    return tf.tile(mask, [batch_size, 1, 1, 1])

# Transformer Block (GPT style)
def gpt_block(x, num_heads, head_size, ff_dim, dropout=0.1):
    input_shape = tf.shape(x)
    batch_size, seq_len = input_shape[0], input_shape[1]

    causal_mask = create_causal_attention_mask(batch_size, seq_len)
    attn_out = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(x, x, attention_mask=causal_mask)
    attn_out = layers.Dropout(dropout)(attn_out)
    x = layers.Add()([x, attn_out])
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    ffn_out = layers.Dense(ff_dim, activation="relu")(x)
    ffn_out = layers.Dense(x.shape[-1])(ffn_out)
    ffn_out = layers.Dropout(dropout)(ffn_out)
    x = layers.Add()([x, ffn_out])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x

# Building the GPT model
def build_gpt(vocab_size, max_len, embed_dim, num_heads, head_size, ff_dim, num_layers, dropout_rate):
    inputs = keras.Input(shape=(max_len,), dtype=tf.int32)

    # Token Embedding
    token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)(inputs)

    # Positional Embedding
    positions = tf.range(start=0, limit=max_len, delta=1)
    pos_emb = layers.Embedding(input_dim=max_len, output_dim=embed_dim)(positions)
    x = token_emb + pos_emb  # Add token + position embeddings

    # Transformer Blocks
    for _ in range(num_layers):
        x = gpt_block(x, num_heads, head_size, ff_dim, dropout_rate)

    # Final dense layer for next-token prediction
    outputs = layers.Dense(vocab_size, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    return model

# Create the model
gpt_model = build_gpt(
    vocab_size=vocab_size,
    max_len=max_len,
    embed_dim=embed_dim,
    num_heads=num_heads,
    head_size=head_size,
    ff_dim=ff_dim,
    num_layers=num_layers,
    dropout_rate=dropout_rate
)

# Compile
gpt_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Model summary
gpt_model.summary()
