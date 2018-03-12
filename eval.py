# -*- coding: utf-8 -*-

import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
import PIL.Image
import piexif
import piexif.helper


def stringify_dict(d):
  newd = {}
  for k, v in d.iteritems():
    if isinstance(v, unicode):
      v = str(v)
    newd[str(k)] = v
  return newd


class Data(object):

  def __init__(self, data_dir, file_list, batch_size):
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


def main():
  # Create data.
  data_dir = "/Users/awaw/me/plant/data"
  batch_size = 5
  test_env = Data(data_dir, "test", batch_size)
  plant_names = [
    "薜荔",
    "腎蕨",
    "蓪草",
    "麻竹",
    "台灣芭蕉",
  ]

  # Load model.
  graph_def = graph_pb2.GraphDef()
  model_file = "/tmp/plant/test_1518554862/checkpoint/freeze.pb"
  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  tf.import_graph_def(graph_def, name="")

  # Define the prediction logic.
  g = tf.get_default_graph()
  img = g.get_tensor_by_name("agent/img:0")
  len_pixel = g.get_tensor_by_name("agent/len_pixel:0")
  len_cm = g.get_tensor_by_name("agent/len_cm:0")
  softmax = g.get_tensor_by_name("agent/Softmax:0")
  pred_prob, pred_class = tf.nn.top_k(softmax, k=3)

  with tf.Session() as sess:
    print("\n--------")
    total_num = 0
    correct_num = 0
    for _ in range(2):
      data = test_env.get()

      feed_dict = {}
      feed_dict[img] = data["img"]
      feed_dict[len_pixel] = data["len_pixel"]
      feed_dict[len_cm] = data["len_cm"]
      run_ops = {}
      run_ops["pred_prob"] = pred_prob
      run_ops["pred_class"] = pred_class
      run_res = sess.run(run_ops, feed_dict=feed_dict)

      for b, fname in enumerate(data["name"]):
        total_num += 1
        top_pred = 0
        pred = run_res["pred_class"][b, top_pred]
        label = data["label"][b]
        pred_str = plant_names[pred]
        label_str = plant_names[label]
        if pred == label:
          correct_num += 1
          correct = "V"
        else:
          correct = "X"
        print("檔名: {}, 正確: {}, 預測: {}, 實際: {}".format(fname, correct, pred_str, label_str))

    print("--------")
    print("準確率: {}% ({}/{})".format(float(correct_num) / total_num * 100, correct_num, total_num))

if __name__ == "__main__":
  main()
