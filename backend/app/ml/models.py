import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, GRU, BatchNormalization, Dropout, Layer
import tensorflow.keras.backend as K

class TemporalAttention(Layer):
    """
    Custom Temporal Attention layer to learn importance weights over the 32 sensor channels.
    Outputs both the attention-weighted context vector and the attention weights.
    """
    def __init__(self, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape is (batch_size, sequence_length, hidden_dim)
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )
        super(TemporalAttention, self).build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch_size, seq_len, hidden_dim)
        # score shape: (batch_size, seq_len, 1)
        score = tf.matmul(inputs, self.W) + self.b
        score = tf.nn.tanh(score)
        
        # weights shape: (batch_size, seq_len, 1)
        weights = tf.nn.softmax(score, axis=1)
        
        # context_vector shape: (batch_size, hidden_dim)
        context_vector = inputs * weights
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        # Return context vector and squeezed weights (batch_size, seq_len)
        return context_vector, tf.squeeze(weights, axis=-1)

    def compute_output_shape(self, input_shape):
        return [
            (input_shape[0], input_shape[-1]),  # Context vector
            (input_shape[0], input_shape[1])    # Attention weights
        ]

    def get_config(self):
        config = super(TemporalAttention, self).get_config()
        return config


def build_baseline_dnn(input_dim=32):
    """
    Baseline Dense Neural Network from reference paper:
    Input (32) -> BN -> Dense(64, ReLU) -> Dense(32, ReLU) -> Dense(16, ReLU) -> Softmax(2)
    """
    inputs = Input(shape=(input_dim,), name='dnn_input')
    x = BatchNormalization()(inputs)
    x = Dense(64, activation='relu', kernel_initializer='glorot_uniform')(x)
    x = Dense(32, activation='relu', kernel_initializer='glorot_uniform')(x)
    x = Dense(16, activation='relu', kernel_initializer='glorot_uniform')(x)
    outputs = Dense(2, activation='softmax', name='dnn_output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='Baseline_DNN')
    return model


def build_hybrid_model(seq_len=32, feature_dim=31):
    """
    Proposed Hybrid CNN-GRU-Attention model:
    Extracts local features from the 32 sensors, processes them as a sequence,
    and applies self-attention to determine sensor feature importance.
    """
    inputs = Input(shape=(seq_len, feature_dim), name='seq_input')
    
    # 1. Local spatial/channel feature extraction across sensors
    x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # 2. Recurrent temporal modeling
    x = GRU(units=64, return_sequences=True)(x)
    
    # 3. Attention pooling over the 32 sensor steps
    context, attn_weights = TemporalAttention(name='attention_layer')(x)
    
    # 4. Dense classification layers
    x = Dense(32, activation='relu')(context)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    class_output = Dense(2, activation='softmax', name='class_output')(x)
    
    # Model returns class probability output AND attention weights (useful for XAI inference)
    model = Model(inputs=inputs, outputs=[class_output, attn_weights], name='Hybrid_CNN_GRU_Attention')
    return model
