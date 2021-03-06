﻿# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os import path
from numpy import *
import numpy as np
import tensorflow as tf
import time

from tensorflow.contrib.timeseries.python.timeseries import estimators as ts_estimators
from tensorflow.contrib.timeseries.python.timeseries import model as ts_model

import pandas as pd
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt


class _LSTMModel(ts_model.SequentialTimeSeriesModel):
  """A time series model-building example using an RNNCell."""

  def __init__(self, num_units, num_features, dtype=tf.float32):

    super(_LSTMModel, self).__init__(
        # Pre-register the metrics we'll be outputting (just a mean here).
        train_output_names=["mean"],
        predict_output_names=["mean"],
        num_features=num_features,
        dtype=dtype)
    self._num_units = num_units
    # Filled in by initialize_graph()
    self._lstm_cell = None
    self._lstm_cell_run = None
    self._predict_from_lstm_output = None

  def initialize_graph(self, input_statistics):
    super(_LSTMModel, self).initialize_graph(input_statistics=input_statistics)
    self._lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self._num_units)
    # Create templates so we don't have to worry about variable reuse.
    self._lstm_cell_run = tf.make_template(
        name_="lstm_cell",
        func_=self._lstm_cell,
        create_scope_now_=True)
    # Transforms LSTM output into mean predictions.
    self._predict_from_lstm_output = tf.make_template(
        name_="predict_from_lstm_output",
        func_=lambda inputs: tf.layers.dense(inputs=inputs, units=self.num_features),
        create_scope_now_=True)

  def get_start_state(self):
    """Return initial state for the time series model."""
    return (
        # Keeps track of the time associated with this state for error checking.
        tf.zeros([], dtype=tf.int64),
        # The previous observation or prediction.
        tf.zeros([self.num_features], dtype=self.dtype),
        # The state of the RNNCell (batch dimension removed since this parent
        # class will broadcast).
        [tf.squeeze(state_element, axis=0)
         for state_element
         in self._lstm_cell.zero_state(batch_size=1, dtype=self.dtype)])

  def _transform(self, data):
    """Normalize data based on input statistics to encourage stable training."""
    mean, variance = self._input_statistics.overall_feature_moments
    return (data - mean) / variance

  def _de_transform(self, data):
    """Transform data back to the input scale."""
    mean, variance = self._input_statistics.overall_feature_moments
    return data * variance + mean

  def _filtering_step(self, current_times, current_values, state, predictions):
    state_from_time, prediction, lstm_state = state
    with tf.control_dependencies(
            [tf.assert_equal(current_times, state_from_time)]):
      transformed_values = self._transform(current_values)
      # Use mean squared error across features for the loss.
      predictions["loss"] = tf.reduce_mean(
          (prediction - transformed_values) ** 2, axis=-1)
      # Keep track of the new observation in model state. It won't be run
      # through the LSTM until the next _imputation_step.
      new_state_tuple = (current_times, transformed_values, lstm_state)
    return (new_state_tuple, predictions)

  def _prediction_step(self, current_times, state):
    """Advance the RNN state using a previous observation or prediction."""
    _, previous_observation_or_prediction, lstm_state = state
    lstm_output, new_lstm_state = self._lstm_cell_run(
        inputs=previous_observation_or_prediction, state=lstm_state)
    next_prediction = self._predict_from_lstm_output(lstm_output)
    new_state_tuple = (current_times, next_prediction, new_lstm_state)
    return new_state_tuple, {"mean": self._de_transform(next_prediction)}

  def _imputation_step(self, current_times, state):
    return state

  def _exogenous_input_step(
          self, current_times, current_exogenous_regressors, state):
    """Update model state based on exogenous regressors."""
    raise NotImplementedError(
        "Exogenous inputs are not implemented for this example.")


if __name__ == '__main__':
  iteration = 1
  n_hidden = 252
  while iteration>0:
    tf.logging.set_verbosity(tf.logging.INFO)
    csv_file_name = 'period_trend_A.csv'
    reader = tf.contrib.timeseries.CSVReader(csv_file_name)
    original_file = 'period_trend_B.csv'
    original_data = pd.read_csv(original_file)

    train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(
        reader, batch_size=4, window_size=100)#batch_size
  #window_size
    estimator = ts_estimators.TimeSeriesRegressor(
        model=_LSTMModel(num_features=1, num_units= n_hidden),#单变量，使用隐层为n_hidden大小的LSTM模型。
        optimizer=tf.train.AdamOptimizer(0.001))

    estimator.train(input_fn=train_input_fn, steps=100)#####################number of steps to train,can be changed to 3000 or more?
    evaluation_input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader)
    evaluation = estimator.evaluate(input_fn=evaluation_input_fn, steps=1)

  # Predict starting after the evaluation
    time_start = time.time()
    (predictions,) = tuple(estimator.predict(
        input_fn=tf.contrib.timeseries.predict_continuation_input_fn(
            evaluation, steps=500)))#########################################num of predicted points
    time_end = time.time()
    time_consume = time_end - time_start
    print ("time_consume = " + str(time_consume))

    observed_times = evaluation["times"][0]
    observed = evaluation["observed"][0, :, :]
    evaluated_times = evaluation["times"][0]
    evaluated = evaluation["mean"][0]
    predicted_times = predictions['times']
    predicted = predictions["mean"]

    plt.figure(figsize=(25, 5))
    predict,pre_t=[],[]       
    for i in range(0,375):
        if i%2==0:
            predict.append(predicted[i])
            pre_t.append(predicted_times[i])
    predicted_lines =  plt.plot(predicted_times[0:375], predicted[0:375], label="prediction", color="blue",linewidth=1)  #plt.scatter(pre_t,predict,marker='o',c='',edgecolors="black")
    alt = array(original_data[['x', 'y']])
    plt.plot(alt[0:2000, 0], alt[0:2000, 1],label="original data",linewidth=1,color="red")
    original=alt[:,1]
    plt.ylabel('LZ')
    plt.xlabel('Time')
    plt.title('2D prediction')
    plt.legend(loc="upper left")

    plt.xlim(xmin=1374)
    plt.xlim(xmax=3375)
    #plt.show()
    itr = str(iteration)
    out_file = 'pre_2D_7/line'+ itr + '.jpg'
    plt.savefig(out_file)
    print ("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    error_pre=predicted-original
    error_f=np.abs(error_pre)
    #print ("ERROR="+error_f+"\n")
    error_avg=np.mean(error_f)
    print (error_avg)
    #print ("\n")
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    iteration -= 1
    #n_hidden += 1
