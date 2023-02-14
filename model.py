import tensorflow as tf

class Encoder (tf.keras.Model):
  def __init__ (self, units, output_dim):
    super(Encoder, self).__init__()
    self.gru1 = tf.keras.layers.GRU(units, recurrent_initializer='glorot_uniform', return_sequences=True, return_state=True)
    self.grus = [tf.keras.layers.GRU(units, recurrent_initializer='glorot_uniform', return_sequences=True, return_state=True) for _ in range(5)]
    self.fc = tf.keras.layers.Dense(output_dim)
    
  def call (self, inputs):
    rnn_outputs, hidden_states = self.gru1(inputs)
    for gru in self.grus:
      rnn_outputs, hidden_states = gru(rnn_outputs)
    outputs = self.fc(rnn_outputs)
    return outputs

class Decoder (tf.keras.Model):
  def __init__ (self, vocab_size, embedding_dim, units, output_dim):
    super(Decoder, self).__init__()
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru1 = tf.keras.layers.GRU(units, recurrent_initializer='glorot_uniform', return_sequences=True, return_state=True)
    self.gru2 = tf.keras.layers.GRU(units, recurrent_initializer='glorot_uniform', return_sequences=True, return_state=True)
    self.fc = tf.keras.layers.Dense(output_dim)

  def call (self, inputs):
    embedded = self.embedding(inputs)
    outputs, hidden_states = self.gru1(embedded)
    outputs, hidden_states = self.gru2(outputs)
    outputs = self.fc(outputs)
    return outputs, hidden_states
  
class JointNet (tf.keras.Model):
  def __init__ (self, inner_dim, vocab_size):
    super(JointNet, self).__init__()
    self.forward_layer = tf.keras.layers.Dense(inner_dim, activation='tanh')
    self.project_layer = tf.keras.layers.Dense(vocab_size)
  
  def call (self, inputs):
    enc_outputs, dec_outputs = inputs
    joint_inputs = tf.expand_dims(enc_outputs, axis=2) + tf.expand_dims(dec_outputs, axis=1)
    outputs = self.forward_layer(joint_inputs)
    outputs = self.project_layer(outputs)
    return outputs

class Transducer (tf.keras.Model):
  def __init__ (self, embedding_dim, units, coder_output_dim, joint_net_inner_dim, vocab_size):
    super(Transducer, self).__init__()
    self.encoder = Encoder(units, coder_output_dim)
    self.decoder = Decoder(vocab_size, embedding_dim, units, coder_output_dim)
    self.joint_net = JointNet(joint_net_inner_dim, vocab_size)
    self.vocab_size = vocab_size
  
  def call (self, inputs):
    enc_inputs, dec_inputs = inputs
    enc_outputs = self.encoder(enc_inputs)
    dec_outputs, dec_states = self.decoder(dec_inputs)
    outputs = self.joint_net([enc_outputs, dec_outputs])
    return outputs


