import json
import os
import threading
import time

import numpy as np
import tensorflow as tf
import PIL.Image
import piexif
import piexif.helper

from agent import simple


def stringify_dict(d):
  newd = {}
  for k, v in d.iteritems():
    if isinstance(v, unicode):
      v = str(v)
    newd[str(k)] = v
  return newd


def saver_restore(saver, cp_dir, sess):
  cp_state = tf.train.get_checkpoint_state(
      cp_dir,
      latest_filename=None)
  paths = cp_state.all_model_checkpoint_paths

  # Start from the latest path
  paths = reversed(paths)

  for restore_dir in paths:
    try:
      saver.restore(sess, restore_dir)
    except tf.errors.NotFoundError as err:
      continue
    return


class Data(object):

  def __init__(self, data_dir, file_list, batch_size):
    super(Data, self).__init__()
    self._data_dir = data_dir
    self.batch_size = batch_size

    with open(os.path.join(self._data_dir, file_list)) as f:
      self._files = f.readlines()
    self._files = [f.strip() for f in self._files]

    self._i = -1

    self.label_map = {
        "a": 0,
        "b": 1,
        "c": 2,
        "d": 3,
        "e": 4,
    }

  def get(self):
    res = self._get_one()
    for k, v in res.iteritems():
      res[k] = np.expand_dims(v, axis=0)

    for b in range(self.batch_size-1):
      datum = self._get_one()
      for k, v in datum.iteritems():
        res[k] = np.append(res[k], np.expand_dims(v, axis=0), axis=0)

    return res

  def _get_one(self):
    self._i += 1
    if self._i >= len(self._files):
      self._i = 0

    fname = self._files[self._i]
    fpath = self._data_dir + "/" + fname
    img = PIL.Image.open(fpath)
    exif_dict = piexif.load(fpath)
    user_comment = piexif.helper.UserComment.load(exif_dict["Exif"][piexif.ExifIFD.UserComment])
    user_comment = json.loads(user_comment)
    user_comment = stringify_dict(user_comment)

    processed_img = self._preprocess_img(img)
    res = {
        "name": fname,
        "label": np.array(self.label_map[user_comment["plantName"]]),
        "len_cm": np.array(user_comment["lengthInCentiMeter"] / 10),
        "len_pixel": np.array(user_comment["lengthInPixel"] / 1000),
        "img": np.array(processed_img),
    }
    return res

  def _preprocess_img(self, img):
    width, height = img.size   # Get dimensions

    small = width
    if height < width:
      small = height
    new_width = small
    new_height = small

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    
    cropped_img = img.crop((left, top, right, bottom))

    scaled_img = cropped_img.resize((128, 128), PIL.Image.ANTIALIAS)

    return scaled_img

  def size(self):
    return len(self._files)


def _log_trainable_variables():
  total_params = 0
  for variable in tf.trainable_variables():
    shape = variable.get_shape()

    var_params = 1
    for dim in shape:
        var_params *= dim.value
    tf.logging.info("%s %s: %s", variable.name, shape, var_params)

    total_params += var_params

  tf.logging.info("total parameters: %s", total_params)


class Logger(object):

  def __init__(self, wid, log_file, period_secs):
    super(Logger, self).__init__()
    self._wid = wid
    self._log_file = log_file
    self._period_secs = period_secs

    self._last_log_time = -1

    self._lock = threading.Lock()
    self._vals = {}

  def set_vals(self, vals):
    with self._lock:
      for k, v in vals.iteritems():
        if v is not None:
          self._vals[k] = v
        else:
          self._vals.pop(k, None)

  def log(self, global_step):
    if time.time() - self._last_log_time < self._period_secs:
      return
    self._last_log_time = time.time()

    self._vals["wid"] = self._wid
    self._vals["time"] = time.time()
    self._vals["step"] = global_step

    with open(self._log_file, "a") as f:
      jstr = json.dumps(self._vals)
      f.write(jstr+"\n")


class Evaluator(object):

  def __init__(self, env, agent, logger, log_prefix, loop_num, period_secs, save_path, sess):
    super(Evaluator, self).__init__()
    self._env = env
    self._agent = agent
    self._logger = logger
    self._log_prefix = log_prefix
    self._loop_num = loop_num
    self._period_secs = period_secs
    self._save_path = save_path
    self._sess = sess

    self._last_eval_time = -1

    if self._save_path != "":
      self._best_score = -9999999999
      save_dir = os.path.dirname(self._save_path)
      if not os.path.exists(save_dir):
        os.makedirs(save_dir)
      self._saver = tf.train.Saver(max_to_keep=1)

  def evaluate(self):
    if time.time() - self._last_eval_time < self._period_secs:
      return
    self._last_eval_time = time.time()

    correct_num = 0
    for i in range(self._loop_num):
      obs = self._env.get()
      test_res = self._agent.test(obs)
      logits = test_res["logits"]

      pred = np.argmax(logits, axis=1)

      correct_num += np.sum(np.equal(pred, obs["label"]))

    correct_percentage = float(correct_num) / (self._loop_num * self._env.batch_size)
    raw_log_vals = {
        "correct_percentage": correct_percentage,
        "loss": test_res["loss"].tolist(),
        "sample_name": obs["name"].tolist(),
        "sample_pred": pred.tolist(),
        "sample_logits": logits.tolist(),
    }
    log_vals = {}
    for k, v in raw_log_vals.iteritems():
      log_vals[self._log_prefix + k] = v
    self._logger.set_vals(log_vals)

    if self._save_path != "":
      if correct_percentage > self._best_score:
        self._best_score = correct_percentage

        global_step = self._agent.step_cnt()
        self._saver.save(self._sess, self._save_path, global_step=global_step)


class PeriodicSaver(object):
  
  def __init__(self, save_path, period_secs):
    super(PeriodicSaver, self).__init__()
    self._save_path = save_path
    self._period_secs = period_secs
    self.session = None

    self._saver = tf.train.Saver()

    self._last_save_time = -1

    save_dir = os.path.dirname(self._save_path)
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)

  def save(self, global_step):
    if time.time() - self._last_save_time < self._period_secs:
      return
    self._last_save_time = time.time()

    self._saver.save(self.session, self._save_path+"/a", global_step=global_step)


def main():
  tf.logging.set_verbosity(tf.logging.INFO)
  wid = "test_{}".format(int(time.time()))
  tf.logging.info("work ID: %s", wid)
  exp_dir = "/tmp/plant/{}".format(wid)

  log_period_secs = 3
  logger = Logger(wid, exp_dir+"/log", log_period_secs)

  data_dir = "/home/denkeni/plant/data"
  data_dir = "/Users/awaw/me/plant/data"
  batch_size = 5
  train_env = Data(data_dir, "train", batch_size)
  test_env = Data(data_dir, "test", batch_size)

  agent = simple.Agent(test_env)

  var_init_op = tf.global_variables_initializer()

  cp_path = exp_dir+"/checkpoint"
  save_period_secs = 3
  saver = PeriodicSaver(cp_path, save_period_secs)

  sess_conf = tf.ConfigProto(log_device_placement=False)
  with tf.Session(config=sess_conf) as sess:
    sess.run([var_init_op])
    _log_trainable_variables()

    train_evaluate_period_secs = 3
    train_evaluator = Evaluator(train_env, agent, logger, "train_", 1, train_evaluate_period_secs, "", sess)
    evaluate_loop_num = int(np.ceil(float(test_env.size()) / test_env.batch_size))
    evaluate_period_secs = 3
    evaluate_save_path = exp_dir+"/eval_cp/a"
    evaluator = Evaluator(test_env, agent, logger, "eval_", evaluate_loop_num, evaluate_period_secs, evaluate_save_path, sess)

    agent.session = sess
    saver.session = sess

    # agent.gen_mlmodel2("/tmp/plant/no_len_cm_1507701074/checkpoint/a-915", "/tmp/plant/no_len_cm_1507701074.mlmodel")

    while True:
      obs = train_env.get()
      #obs["len_cm"] = np.zeros(obs["len_cm"].shape)
      agent.step(obs)

      train_evaluator.evaluate()
      evaluator.evaluate()
      logger.log(agent.step_cnt())
      saver.save(agent.step_cnt())

if __name__ == "__main__":
  main()
