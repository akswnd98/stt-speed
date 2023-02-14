import tensorflow as tf

def nan_to_zero (tensor):
  return tf.where(tf.math.is_nan(tensor), tf.zeros_like(tensor), tensor)

def rnnt_loss (logits, labels, time_lengths, label_lengths):
  log_pr = tf.math.log_softmax(logits, axis=-1)
  pr = pr_loss(log_pr, labels, time_lengths, label_lengths)
  ret = tf.reduce_sum(pr)
  return ret

@tf.custom_gradient
def pr_loss (log_pr, labels, time_lengths, label_lengths):
  LOG_0 = float('-inf')
  batch_size = tf.shape(log_pr)[0]
  max_time_lengths = tf.shape(log_pr)[1]
  max_label_lengths = tf.shape(log_pr)[2]

  def get_truth_log_pr (log_pr, labels):
    labels_one_hot = tf.one_hot(labels, tf.shape(log_pr)[-1], axis=-1, dtype=tf.float32)
    labels_one_hot = labels_one_hot[:, 1: ]
    labels_one_hot = tf.expand_dims(labels_one_hot, axis=1)
    labels_one_hot = tf.repeat(labels_one_hot, tf.shape(log_pr)[1], axis=1)
    ret = tf.reduce_sum(log_pr[:, :, : -1, :] * labels_one_hot, axis=-1)
    ret = tf.concat([
      ret,
      LOG_0 * tf.ones((tf.shape(log_pr)[0], tf.shape(log_pr)[1], 1), dtype=tf.float32)
    ], axis=-1)
    return ret
  
  def get_blank_log_pr (log_pr):
    return log_pr[:, :, :, 0]

  truth_log_pr = get_truth_log_pr(log_pr, labels)
  blank_log_pr = get_blank_log_pr(log_pr)

  def get_alpha (truth_log_pr, blank_log_pr):
    reversed_truth_log_pr = tf.reverse(truth_log_pr, axis=[-1])
    padded_truth_log_pr = tf.pad(
      reversed_truth_log_pr,
      [[0, 0], [0, 0], [tf.shape(reversed_truth_log_pr)[-2] - 1, 0]],
      constant_values=LOG_0
    )
    truth_diag = tf.linalg.diag_part(
      padded_truth_log_pr,
      k=(0, tf.shape(padded_truth_log_pr)[-1] - 1),
      padding_value=LOG_0,
      align='LEFT_RIGHT'
    )
    truth_diag = tf.transpose(truth_diag, perm=[1, 0, 2])

    reversed_blank_log_pr = tf.reverse(blank_log_pr, axis=[-1])
    padded_blank_log_pr = tf.pad(
      reversed_blank_log_pr,
      [[0, 0], [0, 0], [tf.shape(reversed_blank_log_pr)[-2] - 1, 0]],
      constant_values=LOG_0
    )
    blank_diag = tf.linalg.diag_part(
      padded_blank_log_pr,
      k=(0, tf.shape(padded_blank_log_pr)[-1] - 1),
      padding_value=LOG_0,
      align='LEFT_RIGHT'
    )
    blank_diag = tf.concat([
      LOG_0 * tf.ones((tf.shape(blank_diag)[0], tf.shape(blank_diag)[1], 1), dtype=tf.float32),
      blank_diag[:, :, : -1]
    ], axis=-1)
    blank_diag = tf.transpose(blank_diag, perm=[1, 0, 2])

    initial_diag = tf.concat([
      tf.zeros((tf.shape(blank_diag)[1], 1), dtype=tf.float32),
      LOG_0 * tf.ones((tf.shape(blank_diag)[1], tf.shape(blank_diag)[-1] - 1), dtype=tf.float32)
    ], axis=-1)

    def step (a, x):
      t, b = x
      return (
        tf.reduce_logsumexp(
          tf.stack([
              a + t,
              tf.concat([
                LOG_0 * tf.ones((tf.shape(a)[0], 1), dtype=tf.float32),
                a[:, : -1]
              ], axis=-1) + b
            ],
            axis=0
          ), axis=0
        )
      )
    alpha_diag =  tf.concat([
      tf.expand_dims(initial_diag, axis=0),
      tf.scan(step, (truth_diag, blank_diag), initial_diag)
    ], axis=0)
    alpha_diag = tf.transpose(alpha_diag, perm=[1, 2, 0])
    alpha = tf.linalg.diag_part(alpha_diag, k=(0, max_label_lengths - 1))
    alpha = tf.reverse(alpha, axis=[-2])
    alpha = tf.transpose(alpha, perm=[0, 2, 1])
    return alpha

  def get_beta (truth_log_pr, blank_log_pr):
    reversed_truth_log_pr = tf.reverse(truth_log_pr, axis=[-1])
    reversed_truth_log_pr = tf.concat([
      LOG_0 * tf.ones((tf.shape(reversed_truth_log_pr)[0], tf.shape(reversed_truth_log_pr)[1], 1), dtype=tf.float32),
      reversed_truth_log_pr[:, :, 1: ]
    ], axis=-1)
    padded_truth_log_pr = tf.pad(
      reversed_truth_log_pr,
      [[0, 0], [0, 0], [tf.shape(reversed_truth_log_pr)[-2] - 1, 0]],
      constant_values=LOG_0
    )
    truth_diag = tf.linalg.diag_part(
      padded_truth_log_pr,
      k=(0, tf.shape(padded_truth_log_pr)[-1] - 1),
      padding_value=LOG_0,
      align='LEFT_RIGHT'
    )
    truth_diag = tf.transpose(truth_diag, perm=[1, 0, 2])
    truth_diag = truth_diag[: -1]

    reversed_blank_log_pr = tf.reverse(blank_log_pr, axis=[-1])
    reversed_blank_log_pr = tf.concat([
      reversed_blank_log_pr[:, : -1, :],
      LOG_0 * tf.ones((tf.shape(reversed_blank_log_pr)[0], 1, tf.shape(reversed_blank_log_pr)[-1]), dtype=tf.float32)
    ], axis=-2)
    padded_blank_log_pr = tf.pad(
      reversed_blank_log_pr,
      [[0, 0], [0, 0], [tf.shape(reversed_blank_log_pr)[-2] - 1, 0]],
      constant_values=LOG_0
    )
    blank_diag = tf.linalg.diag_part(
      padded_blank_log_pr,
      k=(0, tf.shape(padded_blank_log_pr)[-1] - 1),
      padding_value=LOG_0,
      align='LEFT_RIGHT'
    )
    blank_diag = tf.transpose(blank_diag, perm=[1, 0, 2])
    blank_diag = blank_diag[: -1]

    mask = tf.sequence_mask(
      time_lengths + label_lengths - 2,
      tf.shape(log_pr)[1] + tf.shape(log_pr)[2] - 2,
      dtype=tf.float32
    )
    mask = tf.transpose(mask, perm=[1, 0])

    dp_start_value = tf.gather_nd(
      blank_log_pr,
      indices=tf.stack([time_lengths, label_lengths], axis=-1) - 1,
      batch_dims=1
    )

    initial_diag_mask = tf.one_hot(time_lengths - 1, depth=tf.shape(log_pr)[1])
    initial_diag = tf.expand_dims(dp_start_value, axis=1) * initial_diag_mask + nan_to_zero(LOG_0 * (1.0 - initial_diag_mask))

    def step (a, x):
      m, t, b = x
      a_next = tf.reduce_logsumexp(
        tf.stack([
          a + t,
          tf.concat([
            a[:, 1: ],
            LOG_0 * tf.ones((tf.shape(a)[0], 1), dtype=tf.float32)
          ], axis=-1) + b
        ], axis=0),
        axis=0
      )
      masked_a_next = nan_to_zero(a_next * tf.expand_dims(m, axis=1)) + nan_to_zero(a * tf.expand_dims(1.0 - m, axis=1))
      return masked_a_next

    beta_diag = tf.concat([
      tf.scan(step, (mask, truth_diag, blank_diag), initial_diag, reverse=True),
      tf.expand_dims(initial_diag, axis=0)
    ], axis=0)

    beta_diag = tf.transpose(beta_diag, perm=[1, 2, 0])
    beta = tf.linalg.diag_part(beta_diag, k=(0, tf.shape(log_pr)[2] - 1), padding_value=LOG_0)
    beta = tf.transpose(beta, perm=[0, 2, 1])
    beta = tf.reverse(beta, axis=[-1])

    return beta
  
  time_mask = tf.sequence_mask(time_lengths, tf.shape(log_pr)[1], dtype=tf.float32)
  label_mask = tf.sequence_mask(label_lengths, tf.shape(log_pr)[2], dtype=tf.float32)
  total_mask = tf.expand_dims(time_mask, axis=2) * tf.expand_dims(label_mask, axis=1)

  alpha = get_alpha(truth_log_pr, blank_log_pr)
  alpha = alpha + nan_to_zero((1.0 - total_mask) * LOG_0)
  beta = get_beta(truth_log_pr, blank_log_pr)
  beta = beta + nan_to_zero((1.0 - total_mask) * LOG_0)

  indices = tf.concat([
    tf.expand_dims(tf.range(0, tf.shape(log_pr)[0]), axis=1),
    tf.stack([
      time_lengths,
      label_lengths - 1
    ], axis=-1),
  ], axis=-1)
  
  beta_mask = tf.scatter_nd(
    indices,
    tf.ones(tf.shape(indices)[0], tf.float32),
    (tf.shape(log_pr)[0], tf.shape(log_pr)[1] + 1, tf.shape(log_pr)[2])
  )
  beta_mask = 1.0 - beta_mask
  beta = nan_to_zero(tf.pad(beta, [[0, 0], [0, 1], [0, 0]], constant_values=LOG_0) * beta_mask)

  total_mask = tf.expand_dims(total_mask, axis=-1)

  total_log_pr = beta[:, 0, 0]

  def grad (upstream):
    blank_grads = \
      alpha + beta[:, 1: , :] \
      - tf.reshape(total_log_pr, shape=(tf.shape(total_log_pr)[0], 1, 1))
    truth_grads = \
      alpha + tf.pad(
        beta[:, : -1, 1: ],
        [[0, 0], [0, 0], [0, 1]],
        constant_values=LOG_0
      ) \
      - tf.reshape(total_log_pr, shape=(tf.shape(total_log_pr)[0], 1, 1))
    blank_one_hot = tf.one_hot(tf.zeros_like(labels, dtype=tf.int32), tf.shape(log_pr)[-1], dtype=tf.float32)
    blank_one_hot = tf.expand_dims(blank_one_hot, axis=1)
    blank_one_hot = tf.repeat(blank_one_hot, tf.shape(log_pr)[1], axis=1)
    blank_grads = tf.exp(tf.expand_dims(blank_grads, axis=-1) + log_pr) * blank_one_hot
    truth_one_hot = tf.one_hot(labels[:, 1: ], tf.shape(log_pr)[-1], dtype=tf.float32)
    truth_one_hot = tf.concat([truth_one_hot, tf.zeros((tf.shape(log_pr)[0], 1, tf.shape(log_pr)[-1]), dtype=tf.float32)], axis=-2)
    truth_one_hot = tf.expand_dims(truth_one_hot, axis=1)
    truth_one_hot = tf.repeat(truth_one_hot, tf.shape(log_pr)[1], axis=1)
    truth_grads = tf.exp(tf.expand_dims(truth_grads, axis=-1) + log_pr) * truth_one_hot

    grads = blank_grads + truth_grads

    return (
      [
        tf.reshape(-upstream, shape=(tf.shape(upstream)[0], 1, 1, 1))
        * grads * total_mask
      ] +
      [None] * 3
    )

  return -total_log_pr, grad
