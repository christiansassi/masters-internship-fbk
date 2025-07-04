from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import ( # type: ignore
    Input, Conv1D, MaxPooling1D, UpSampling1D,
    LSTM, RepeatVector, TimeDistributed,
    Dense, Dropout, BatchNormalization, Concatenate
)

from tensorflow.keras.optimizers import Adam # type: ignore

def autoencoder_model(
    x_shape: tuple,
    lstm_units: int = 64,
    conv_filters: int = 32,
    latent_dim: int = 16,
    dropout_rate: float = 0.1,
    learning_rate: float = 1e-3
) -> Model:
    
    timesteps, features = x_shape
    inp = Input(shape=(timesteps, features))

    # --- Encoder ---
    x = Conv1D(conv_filters, kernel_size=3, padding="causal", activation="relu")(inp)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2, padding="same")(x)
    x = Dropout(dropout_rate)(x)

    x = LSTM(lstm_units, return_sequences=False)(x)
    x = Dropout(dropout_rate)(x)

    # Bottleneck
    latent = Dense(latent_dim, activation="relu", name="latent_vector")(x)

    # --- Decoder ---
    x = RepeatVector(timesteps // 2)(latent)
    x = LSTM(lstm_units, return_sequences=True)(x)
    x = Dropout(dropout_rate)(x)

    x = UpSampling1D(size=2)(x)
    x = Conv1D(conv_filters, kernel_size=3, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)

    # skip‑connection from early conv output
    skip = Conv1D(conv_filters, kernel_size=1, padding="same")(inp)
    x = Concatenate()([x, skip[:, :timesteps, :conv_filters]])

    # Final reconstruction
    out = Conv1D(features, kernel_size=1, activation="linear", padding="same", name="reconstruction")(x)

    model = Model(inputs=inp, outputs=out, name="swat_autoencoder")
    model.compile(optimizer=Adam(learning_rate), loss="mse")
    return model


def threshold_model(
    input_shape: int = 5,
    hidden_dims: tuple = (64, 32),
    dropout_rate: float = 0.1,
    learning_rate: float = 1e-3
) -> Model:

    inp = Input(shape=(input_shape,), name="error_features")

    x = Dense(hidden_dims[0], activation="relu")(inp)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(hidden_dims[1], activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    out = Dense(1, activation="linear", name="threshold")(x)

    model = Model(inputs=inp, outputs=out, name="swat_threshold_regressor")
    model.compile(
        optimizer=Adam(learning_rate),
        loss="mse",
        metrics=["mae"]
    )
    return model
