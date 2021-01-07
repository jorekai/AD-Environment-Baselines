from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from pandas import DataFrame

import config


def plot_series(series,
                name="______",
                actions=None):
    plt.figure(figsize=(15, 7))
    if actions:
        plt.plot(series.index,
                 actions,
                 label="Actions",
                 linestyle="solid")
    plt.plot(series.index,
             series["anomaly"],
             label="True Label",
             linestyle="dotted",
             color="red",
             alpha=.5)
    plt.plot(series.index,
             series["value"],
             label="Series",
             linestyle="solid",
             color="blue",
             linewidth=.5)
    plt.title(f"Series: {name}")
    plt.xlabel('Timestamp')
    plt.ylabel('Value')
    plt.legend()
    plt.show()


def plot_series_scatter(series):
    fig, ax = plt.subplots(figsize=(19.20, 10.80))
    sb.scatterplot(data=series,
                   x="indices",
                   y="value",
                   hue="anomaly",
                   s=(series.anomaly * 8 + 1),
                   ax=ax,
                   color="red", )
    handles, labels = ax.get_legend_handles_labels()
    labels[0] = "Normal"
    labels[1] = "Anomaly"
    ax.legend(handles=handles[0:], labels=labels[0:], loc=8,
              ncol=1, bbox_to_anchor=[0.5, -.3, 0, 0])
    plt.xlabel("Timestamp")
    plt.ylabel("Value")
    plt.show()
    return fig


def save_plot_high_quality(plot, name):
    plot.savefig(config.ROOT_DIR + config.FIGURE_PATH + name, format='jpg', dpi=800)


def save_plot_normal_quality(plot, name):
    plot.savefig(config.ROOT_DIR + config.FIGURE_PATH + name, format='jpg', dpi=400)


def plot_training(rewards):
    figure = plt.figure(figsize=(15, 7))
    data = pd.DataFrame(data=rewards, columns=["reward"])
    plt.plot(data.index,
             data["reward"],
             label="Series",
             linestyle="solid",
             color="blue",
             linewidth=2)
    plt.title(f"Rewards")
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()
    save_plot_normal_quality(figure, config.REWARDS_TRAINING)


def plot_evaluation_series(series, actions):
    plt.figure(figsize=(15, 7))
    plt.plot(series.index, actions, label="Actions", linestyle="solid")
    plt.plot(series.index, series["anomaly"], label="True Label", linestyle="dotted")
    plt.plot(series.index, series["value"], label="Series", linestyle="dashed")
    plt.legend()
    plt.ylabel('Reward Sum')
    plt.show()


def plot(series: DataFrame, actions: List):
    plt.figure(figsize=(12, 7))
    plt.plot(series.index, actions, label="Actions", linestyle="solid")
    for col in series.columns:
        plt.plot(series.index, series[col], label=col, linestyle="dashed")
    plt.legend()
    plt.ylabel('Scaled Values + Action')
    plt.xlabel("Series Index")
    plt.show()


def plot_confusion_matrix(stats, title='Hit vs. Miss Distribution', cmap=plt.cm.Blues):
    cm = np.array([[stats["TP"], stats["TN"]], [stats["FP"], stats["FN"]]])
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Anomaly', 'Normal'])
    plt.yticks(tick_marks, ['Hit', 'Miss'])
    plt.tight_layout()

    width, height = cm.shape

    for x in range(width):
        for y in range(height):
            plt.annotate(str(cm[x][y]), xy=(y, x),
                         horizontalalignment='center',
                         verticalalignment='center')
    plt.ylabel('Predictions')
    plt.xlabel('Timeseries Labels')
