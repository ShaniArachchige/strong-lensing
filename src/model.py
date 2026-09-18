"""
BayesFlow model definition for strong-lens parameter inference.
"""

import os
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers
import bayesflow as bf


@keras.saving.register_keras_serializable(package="lensflow")
class LensCNN(bf.networks.SummaryNetwork):
    def __init__(self, summary_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.summary_dim = summary_dim
        self.conv_stack = keras.Sequential([
            layers.Conv2D(32, 3, strides=1, padding="same", activation="relu"),
            layers.Conv2D(32, 3, strides=2, padding="same", activation="relu"),
            layers.Conv2D(64, 3, strides=1, padding="same", activation="relu"),
            layers.Conv2D(64, 3, strides=2, padding="same", activation="relu"),
            layers.Conv2D(128, 3, strides=1, padding="same", activation="relu"),
            layers.Conv2D(128, 3, strides=2, padding="same", activation="relu"),
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dense(summary_dim),
        ])

    def call(self, x, **kwargs):
        return self.conv_stack(x, training=kwargs.get("training", False))

    def get_config(self):
        config = super().get_config()
        config.update({"summary_dim": self.summary_dim})
        return config


def build_adapter():
    return (
        bf.Adapter()
        .rename("images", "summary_variables")
        .rename("parameters", "inference_variables")
    )


def build_networks(summary_dim=64, coupling_depth=6):
    summary_network = LensCNN(summary_dim=summary_dim)
    inference_network = bf.networks.CouplingFlow(depth=coupling_depth)
    return summary_network, inference_network


def build_approximator(summary_dim=64, coupling_depth=6, learning_rate=1e-4):
    adapter = build_adapter()
    summary_network, inference_network = build_networks(summary_dim, coupling_depth)

    approximator = bf.approximators.ContinuousApproximator(
        summary_network=summary_network,
        inference_network=inference_network,
        adapter=adapter,
    )
    optimizer = keras.optimizers.AdamW(learning_rate=learning_rate)
    approximator.compile(optimizer=optimizer)
    return approximator
