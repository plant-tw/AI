import numpy as np
import tensorflow as tf
import sonnet as snt
from coremltools.proto import Model_pb2
from coremltools.models import neural_network
from coremltools.models import datatypes
from coremltools.models import utils


class Agent(object):

  def __init__(self, env):
    super(Agent, self).__init__()
    self.session = None

    with tf.variable_scope("agent"):
      self._step_cnt_op = tf.get_variable("step_cnt", shape=(), dtype=tf.int64, initializer=tf.constant_initializer(0))
      self._incr_step_cnt_op = tf.assign_add(self._step_cnt_op, 1)

      self._batch_size = env.batch_size
      self._num_classes = len(env.label_map)
      data = env.get()
      self._img_ph = tf.placeholder(name="img", shape=data["img"].shape, dtype=tf.float32)
      self._len_pixel_ph = tf.placeholder(name="len_pixel", shape=data["len_pixel"].shape, dtype=tf.float32)
      self._len_cm_ph = tf.placeholder(name="len_cm", shape=data["len_cm"].shape, dtype=tf.float32)
      self._labels_ph = tf.placeholder(name="labels", shape=(self._batch_size, self._num_classes), dtype=tf.float32)

      self._logits_op = self._wire(self._img_ph, self._len_pixel_ph, self._len_cm_ph)

      cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
          labels=self._labels_ph, logits=self._logits_op)
      self._loss_op = tf.reduce_mean(cross_entropy)

      learning_rate = 1e-3
      optimizer = tf.train.RMSPropOptimizer(learning_rate)
      self._optimize_op = optimizer.minimize(self._loss_op)

      self._softmax_op = tf.nn.softmax(self._logits_op)

  def step(self, obs):
    self._incr_step_cnt()

    labels = self._soft_classification(obs["label"])
    feed_dict = {
        self._img_ph: obs["img"],
        self._len_pixel_ph: obs["len_pixel"],
        self._len_cm_ph: obs["len_cm"],
        self._labels_ph: labels, 
    }
    run_ops = {
        "optimize": self._optimize_op,
    }
    run_res = self.session.run(run_ops, feed_dict=feed_dict)

  def test(self, obs):
    labels = self._soft_classification(obs["label"])
    feed_dict = {
        self._img_ph: obs["img"],
        self._len_pixel_ph: obs["len_pixel"],
        self._len_cm_ph: obs["len_cm"],
        self._labels_ph: labels, 
    }
    run_ops = {
        "logits": self._logits_op,
        "loss": self._loss_op,
    }
    run_res = self.session.run(run_ops, feed_dict=feed_dict)
    return run_res

  def _soft_classification(self, label):
    correct_val = 0.9
    wrong_val = (1 - correct_val) / (self._num_classes - 1)

    labels = wrong_val * np.ones(self._labels_ph.shape)
    for batch, lb in enumerate(label):
      labels[batch][lb] = correct_val

    return labels

  def _wire(self, img, len_pixel, len_cm):
    img_embedding = self._process_img(img)
    embedding  = tf.concat([img_embedding, tf.expand_dims(len_pixel, axis=1), tf.expand_dims(len_cm, axis=1)], axis=1)

    output_size = (32,)
    hidden = embedding
    for i in range(len(output_size)):
      hidden = snt.Linear(output_size=output_size[i])(hidden)
      hidden = tf.tanh(hidden)

    hidden = snt.Linear(output_size=self._num_classes)(hidden)

    return hidden

  def _process_img(self, img):
    with tf.variable_scope("process_img"):
      img_float32 = tf.cast(img, tf.float32)
      img_float32 = tf.div(img_float32, 255)

      output_channels = (16, 32, 64, 32)
      kernel_shape = (3, 3, 3, 3)
      rate = (1, 2, 4, 8)
      hidden = img_float32
      for i in range(len(output_channels)):
        hidden = snt.Conv2D(output_channels[i], kernel_shape[i], rate=rate[i])(hidden)
        hidden = tf.tanh(hidden)

      # Max across width and height.
      hidden = tf.reduce_max(hidden, axis=1)
      # the width has been reduced, so height's axis is still 1.
      hidden = tf.reduce_max(hidden, axis=1)

      return hidden

  def _incr_step_cnt(self):
    self.session.run(self._incr_step_cnt_op)

  def step_cnt(self):
    run_res = self.session.run([self._step_cnt_op])
    return run_res[0]

  def gen_mlmodel2(self, filepattern, out_file):
    reader = tf.train.NewCheckpointReader(filepattern)

    img_shape = self._img_ph.shape.as_list()  # b, h, w, c
    input = [
        ("img", datatypes.Array(img_shape[3], img_shape[1], img_shape[2])),
        ("len_pixel", datatypes.Array(1)),
        ("len_cm", datatypes.Array(1)),
    ]
    output = [
        ("label", datatypes.Array(self._num_classes)),
    ]
    builder = neural_network.NeuralNetworkBuilder(input, output)

    # CoreML has a bug:
    # https://github.com/apple/coremltools/pull/45
    #builder.add_elementwise(name="div_255", input_names=["img"], output_name="div_255", mode="MULTIPLY", alpha=1.0 / 255)
    spec_layer = builder.nn_spec.layers.add()
    spec_layer.name = "div_255"
    spec_layer.input.append("img")
    spec_layer.output.append("div_255")
    spec_layer.multiply.MergeFromString("")
    spec_layer.multiply.alpha = 1.0 / 255

    output_channels = (16, 32, 64, 32)
    kernel_shape = (3, 3, 3, 3)
    rate = (1, 2, 4, 8)
    hidden = {"name": "div_255", "channels": img_shape[3]}
    for i in range(len(output_channels)):
      conv_name = "agent/process_img/conv_2d_{}".format(i)
      if i == 0:
        conv_name = "agent/process_img/conv_2d"
      builder.add_convolution(
          name=conv_name,
          input_name=hidden["name"],
          output_name=conv_name,
          is_deconv=False,
          kernel_channels=hidden["channels"],
          output_channels=output_channels[i],
          height=kernel_shape[i],
          width=kernel_shape[i],
          dilation_factors=(rate[i], rate[i]),
          has_bias=True,
          W=reader.get_tensor(conv_name+"/w"),
          b=reader.get_tensor(conv_name+"/b"),
          groups=1,
          border_mode="same",
          stride_height=1,
          stride_width=1)
      hidden["name"] = conv_name
      hidden["channels"] = output_channels[i]

      tanh_name = "agent/process_img/tanh_{}".format(i)
      builder.add_activation(
          name=tanh_name,
          non_linearity="TANH",
          input_name=hidden["name"],
          output_name=tanh_name)
      hidden["name"] = tanh_name

    reduce_name = "reduce_wh"
    builder.add_reduce(name=reduce_name, input_name=hidden["name"], output_name=reduce_name, axis="HW", mode="max")
    hidden["name"] = reduce_name

    embedding_name = "embedding"
    builder.add_elementwise(name=embedding_name, input_names=[hidden["name"], "len_pixel", "len_cm"], output_name=embedding_name, mode="CONCAT")
    hidden["name"] = embedding_name
    hidden["channels"] += 2

    output_size = (32,)
    linear_idx = -1
    for opsz in output_size:
      linear_idx += 1
      linear_name = "agent/linear_{}".format(linear_idx)
      if linear_idx == 0:
        linear_name = "agent/linear"
      # CoreML's linear product expects the weights to be of shape
      # (output_size, input_size), but Tensorflow gives (input_size, output_size)
      builder.add_inner_product(
          name=linear_name,
          has_bias=True,
          input_name=hidden["name"],
          input_channels=hidden["channels"],
          output_name=linear_name,
          output_channels=opsz,
          W=np.swapaxes(reader.get_tensor(linear_name+"/w"), 0, 1),
          b=reader.get_tensor(linear_name+"/b"))
      hidden["name"] = linear_name
      hidden["channels"] = opsz

      tanh_name = "agent/tanh_{}".format(i)
      builder.add_activation(
          name=tanh_name,
          non_linearity="TANH",
          input_name=hidden["name"],
          output_name=tanh_name)
      hidden["name"] = tanh_name

    linear_idx += 1
    linear_name = "agent/linear_{}".format(linear_idx)
    if linear_idx == 0:
      linear_name = "agent/linear"
    builder.add_inner_product(
        name=linear_name,
        has_bias=True,
        input_name=hidden["name"],
        input_channels=hidden["channels"],
        output_name=linear_name,
        output_channels=self._num_classes,
        W=np.swapaxes(reader.get_tensor(linear_name+"/w"), 0, 1),
        b=reader.get_tensor(linear_name+"/b"))
    hidden["name"] = linear_name
    hidden["channels"] = opsz

    builder.add_softmax(name="softmax", input_name=hidden["name"], output_name="label")

    utils.save_spec(builder.spec, out_file)
