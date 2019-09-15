from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os import path

import numpy
import tensorflow as tf
import time

from tensorflow.contrib.timeseries.python.timeseries import estimators as ts_estimators
from tensorflow.contrib.timeseries.python.timeseries import model as ts_model
import matplotlib
import pandas as pd
matplotlib.use("agg")
import matplotlib.pyplot as plt
from scipy.interpolate import spline

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
  itration1 = 100
  n_step = 500
  while itration1 > 0:
        tf.logging.set_verbosity(tf.logging.INFO)
        csv_file_name = '/home/bjyang/Python-LSTM/data/pab_1-100tran_1.csv'
        original_file = '/home/bjyang/Python-LSTM/data/pab_1-100.csv'
        original_data = pd.read_csv(original_file)

        reader = tf.contrib.timeseries.CSVReader(
            csv_file_name,
            column_names=((tf.contrib.timeseries.TrainEvalFeatures.TIMES,)
                            + (tf.contrib.timeseries.TrainEvalFeatures.VALUES,) * 6001))####5001
        train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(
            reader, batch_size=4, window_size=32)

        estimator = ts_estimators.TimeSeriesRegressor(######################
            model=_LSTMModel(num_features=6001, num_units = 128),########5001
            optimizer=tf.train.AdamOptimizer(0.001))

        estimator.train(input_fn=train_input_fn, steps= n_step)################
        evaluation_input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader)
        evaluation = estimator.evaluate(input_fn = evaluation_input_fn, steps=1)
  # Predict starting after the evaluation

        (predictions,) = tuple(estimator.predict(
            input_fn=tf.contrib.timeseries.predict_continuation_input_fn(
            evaluation, steps=1)))

        predicted_times = predictions['times']
        predicted = predictions["mean"]

        plt.figure(figsize=(15, 5))
        alt1 = numpy.array(original_data[['x', 'y100']])
        plt.plot(alt1[:, 0], alt1[:, 1],label='original data',color='green',linewidth=1)
        predicted_lines = plt.plot(alt1[:, 0], predicted[0,:], label="prediction", color="r",linewidth=1)#####
        plt.ylabel('LZ')
        plt.xlabel('Time')
        plt.title('3D prediction')
        plt.legend(loc="upper left")

        itr = str(itration1)
        out_file = '/home/bjyang/Python-LSTM/predict_result/pre10/line2D_' + itr + '.jpg'
        plt.savefig(out_file,dpi=200)
        itration1 -= 1
        n_step += 100