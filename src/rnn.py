# pyright: reportPrivateImportUsage=false

import json
from typing import Tuple

import click
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

Carry = jax.Array


class RNNCell(nn.RNNCellBase):
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, carry: Carry, x: jax.Array):
        h = carry

        jax.debug.print(f"x.shape {x.shape}, h.shape {h.shape}")
        h = nn.Dense(self.hidden_dim)(x) + nn.Dense(self.hidden_dim)(h)
        h = nn.activation.sigmoid(h)

        y = nn.Dense(self.output_dim)(h)
        y = nn.activation.sigmoid(y)

        return h, y

    @nn.nowrap
    def initialize_carry(self, key: jax.random.KeyArray, input_shape: Tuple[int, ...]) -> Carry:
        initializer = nn.initializers.uniform()
        return initializer(key, (self.hidden_dim, 1))

    @property
    def num_feature_axes(self) -> int:
        return 2


def get_model():
    return nn.RNN(RNNCell(10, 2))


def get_initial_params(model: nn.RNN, key: jax.random.KeyArray, input_shape: tuple[int, int, int]):
    return model.init(key, jnp.ones(input_shape))


def get_train_state(input_shape: tuple[int, int, int]):
    model = get_model()
    params = get_initial_params(model, jax.random.PRNGKey(0), input_shape)
    tx = optax.adam(1e-2)

    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    return state


@jax.jit
def train_step(state: train_state.TrainState, batch: jax.Array):
    x = batch[:-1]
    y_target = batch[1:]

    def loss_fn(params):
        y_pred = state.apply_fn({"params": params}, x)
        loss = jnp.mean((y_pred - y_target) ** 2)

        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params)
    state = state.apply_gradients(grads=grad)

    metrics = {"loss": loss}

    return state, metrics


@click.command()
@click.option("--type", type=click.Choice(["circle", "eight", "double"]), required=True)
def train(type: str):
    n_epochs = 600

    with open(f"{type}.json", "r") as f:
        data = json.load(f)

    trajectories = [jnp.array(trajectory).reshape(-1, 2, 1) for trajectory in data]
    points = jnp.concatenate(trajectories, axis=0)

    (min_x, min_y), (max_x, max_y) = jnp.min(points, axis=0), jnp.max(points, axis=0)
    jax.debug.print("min_x {min_x}, min_y {min_y}, max_x {max_x}, max_y {max_y}", min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y)

    shift = jnp.array([min_x, min_y])
    scale = jnp.array([max_x - min_x, max_y - min_y])
    for i in range(len(trajectories)):
        trajectories[i] = 0.1 + 0.8 * ((trajectories[i] - shift) / scale)

    trajectories = [trajectory[:: len(trajectory) // 100] for trajectory in trajectories]
    max_length = max([len(trajectory) for trajectory in trajectories])

    input_shape = (1, max_length, 2)  # (batch_size, sequence_length, input_dim)
    state = get_train_state(input_shape)

    for epoch in range(n_epochs):
        total_loss = 0.0
        for trajectory in trajectories:
            state, metrics = train_step(state, trajectory)
            total_loss += metrics["loss"]
        jax.debug.print(f"Epoch {epoch}, Loss: {total_loss / len(trajectories)}")


@click.command()
def predict():
    pass


@click.group()
def rnn():
    pass


rnn.add_command(train)
rnn.add_command(predict)
