import matplotlib.pyplot as plt
import tensorflow as tf
from slidingwindow_generator.slidingwindow_generator import \
    SlidingWindowGenerator


class PlotableWindowGenerator(SlidingWindowGenerator):
    # Taken from https://www.tensorflow.org/tutorials/structured_data/time_series
    def plot(
        self,
        plot_col,
        model=None,
        max_subplots=3,
        sequence_stride=1,
        shuffle=True,
        batch_size=32,
    ):
        inputs, labels = self.example(
            sequence_stride=sequence_stride, shuffle=shuffle, batch_size=batch_size
        )
        plt.figure(figsize=(12, 8 * max_subplots // 3))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f"{plot_col} [normed]")
            plt.plot(
                self.input_indices,
                inputs[n, :, plot_col_index],
                label="Inputs",
                marker=".",
                zorder=-10,
            )

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(
                self.label_indices,
                labels[n, :, label_col_index],
                edgecolors="k",
                label="Labels",
                c="#2ca02c",
                s=64,
            )

            if model is not None:
                # predictions = model(inputs)
                predictions = model.predict(inputs)
                predictions = tf.reshape(predictions, labels.shape)
                plt.scatter(
                    self.label_indices,
                    predictions[n, :, label_col_index],
                    marker="X",
                    edgecolors="k",
                    label="Predictions",
                    c="#ff7f0e",
                    s=64,
                )

            if n == 0:
                plt.legend()

        plt.xlabel("Time")
