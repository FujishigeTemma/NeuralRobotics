import glob
import json
import os

import click
import jax
import jax.numpy as jnp
import jax.random as random
import optax
import rerun as rr
from jax.nn.initializers import uniform
from jax.tree_util import tree_map

import wandb

Params = tuple[jax.Array, ...]
Carry = jax.Array


class RNNCell:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def init(self, key: random.KeyArray) -> Params:
        keys = random.split(key, 3)
        Wxh = random.normal(keys[0], (self.hidden_dim, self.input_dim)) * 0.01
        Whh = random.normal(keys[1], (self.hidden_dim, self.hidden_dim)) * 0.01
        bh: jax.Array = jnp.zeros((self.hidden_dim, 1))
        Why = random.normal(keys[2], (self.output_dim, self.hidden_dim)) * 0.01
        by: jax.Array = jnp.zeros((self.output_dim, 1))
        h0: Carry = jnp.zeros((self.hidden_dim, 1))
        return (Wxh, Whh, bh, Why, by, h0)

    def __call__(self, params: Params, carry: Carry, x) -> tuple[Carry, jax.Array]:
        (Wxh, Whh, bh, Why, by, h0) = params
        h = carry

        h = jax.nn.sigmoid(jnp.dot(Wxh, x) + jnp.dot(Whh, h) + bh)
        y = jax.nn.sigmoid(jnp.dot(Why, h) + by)

        return h, y

    def initialize_carry(self, key) -> Carry:
        initializer = uniform()
        return initializer(key, (self.hidden_dim, 1))


class RNN:
    def __init__(self, cell: RNNCell):
        self.cell = cell

    def __call__(self, params: Params, inputs, init_carry: Carry | None = None, init_key: random.KeyArray | None = None) -> tuple[jax.Array, jax.Array]:
        if init_carry is None:
            if init_key is None:
                raise ValueError("Either init_carry or init_key must be provided")
            carry = self.cell.initialize_carry(init_key)
        else:
            carry = init_carry
        carries = []
        outputs = []
        for t in range(inputs.shape[0] - 1):
            carry, output = self.cell(params, carry, inputs[t])
            carries.append(carry)
            outputs.append(output)
        return jnp.array(carries), jnp.array(outputs)


@jax.jit
def sigmoid_derivative(x):
    return jax.nn.sigmoid(x) * (1 - jax.nn.sigmoid(x))


@jax.jit
def compute_grads(params: Params, x, carry, y_pred, y_target, _lambda):
    (Wxh, Whh, bh, Why, by, h0) = params
    h = carry

    2 * (y_pred - y_target)

    y_raw = jnp.dot(Why, h) + by
    sigmoid_derivative(y_raw) * h

    return (dWxh, dWhh, dbh, dWhy, dby, dh0), dh


@jax.jit
def update_params(params: Params, grads, learning_rate):
    return tree_map(lambda p, g: p - learning_rate * g, params, grads)


def save_params(params: Params, path: str):
    jnp.savez(path, *params)


def load_params(path: str) -> Params:
    with jnp.load(path) as data:
        return tuple([jnp.array(v) for v in data.values()])


@click.command()
@click.option("--type", required=True)
@click.option("--seed", required=True, default=0)
@click.option("--n-epochs", required=True, type=int)
@click.option("--learning-rate", required=True, type=float)
@click.option("---lambda", required=True, type=float)
@click.option("--learn-h0", is_flag=True)
@click.option("--output", required=True)
@click.option("--resume", is_flag=True)
def train(type: str, seed: int, n_epochs: int, learning_rate: float, _lambda: float, learn_h0: bool, output: str, resume: bool):
    input_dim, hidden_dim, output_dim = 2, 10, 2
    cell = RNNCell(input_dim, hidden_dim, output_dim)
    rnn = RNN(cell)

    key = random.PRNGKey(seed)
    if resume:
        saved_epochs = glob.glob(os.path.join(output, type, f"rnn_{type}_*.npz"))
        latest_epoch = max([int(f.split("_")[-1].split(".")[0]) for f in saved_epochs])
        params = load_params(os.path.join(output, type, f"rnn_{type}_{latest_epoch}.npz"))
    else:
        latest_epoch = 0
        key, *subkeys = random.split(key, 3)
        params = cell.init(subkeys[0])
        if learn_h0:
            params = (params[0], params[1], params[2], params[3], params[4], rnn.cell.initialize_carry(subkeys[1]))

    clipping_threshold = 1.0

    config = {
        "type": type,
        "seed": seed,
        "n_epochs": n_epochs,
        "lambda": _lambda,
        "clipping_threshold": clipping_threshold,
        "learning_rate": learning_rate,
        "learn_h0": learn_h0,
    }

    wandb.init(project=f"rnn_scratch_{type}", config=config)

    with open(f"{type}.json", "r") as f:
        data = json.load(f)

    trajectories = [jnp.array(trajectory).reshape(-1, 2, 1) for trajectory in data]
    trajectories = trajectories[:1]

    points = jnp.concatenate(trajectories, axis=0)
    (min_x, min_y), (max_x, max_y) = jnp.min(points, axis=0), jnp.max(points, axis=0)

    shift = jnp.array([min_x, min_y])
    scale = jnp.array([max_x - min_x, max_y - min_y])
    for i in range(len(trajectories)):
        trajectories[i] = 0.1 + 0.8 * ((trajectories[i] - shift) / scale)

    trajectories = [trajectory[:: len(trajectory) // 100] for trajectory in trajectories]

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=50 * len(trajectories),
        decay_steps=n_epochs * len(trajectories),
        end_value=0.0,
    )
    optimizer = optax.chain(optax.clip_by_global_norm(clipping_threshold), optax.adam(learning_rate=schedule))
    opt_state = optimizer.init(params)

    for epoch in range(latest_epoch, n_epochs):
        total_loss = 0.0
        for trajectory in trajectories:
            x = jnp.array(trajectory[:-1])
            y_target = jnp.array(trajectory[1:])

            if learn_h0:
                carries, y_preds = rnn(params, x, init_carry=params[5])
            else:
                key, subkey = random.split(key)
                carries, y_preds = rnn(params, x, init_key=subkey)

            total_grads = None
            T = len(y_preds)
            for t in reversed(range(T)):
                x_t, carry, y_pred, y_t = x[t], carries[t], y_preds[t], y_target[t]
                total_loss += jnp.mean((y_pred - y_t) ** 2)

                grads = compute_grads(params, x_t, carry, y_pred, y_t, _lambda)
                grads = tree_map(lambda g: jnp.clip(g, -clipping_threshold, clipping_threshold), grads)
                if total_grads is None:
                    total_grads = grads
                else:
                    total_grads = tree_map(lambda g1, g2: g1 + g2, total_grads, grads)

            if total_grads is None:
                raise ValueError("total_grads is None because the trajectory is empty")

            updates, opt_state = optimizer.update(total_grads, opt_state, params)
            params: Params = optax.apply_updates(params, updates)  # type: ignore -- optax.apply_updates returns a compatible type

        jax.debug.print("Epoch {epoch}, Loss: {loss}", epoch=epoch + 1, loss=total_loss / len(trajectories))
        wandb.log(
            {
                "loss": total_loss / len(trajectories),
                "Why": jnp.mean(params[0]),
                "Whh": jnp.mean(params[1]),
                "by": jnp.mean(params[2]),
                "Wxh": jnp.mean(params[3]),
                "bh": jnp.mean(params[4]),
                "h0": jnp.mean(params[5]),
            },
        )

        if (epoch + 1) % 100 == 0:
            save_params(params, os.path.join(output, type, f"rnn_{type}_{epoch+1}.npz"))


@click.command()
@click.option("--type", required=True)
@click.option("--seed", required=True, default=0)
@click.option("--epoch", required=True, type=int)
@click.option("--use-input", is_flag=True)
@click.option("--use-h0", is_flag=True)
@click.option("--n-samples", type=int)
@click.option("--n-steps", type=int)
@click.option("--output", required=True)
def predict(type: str, seed: int, epoch: int, use_input: bool, use_h0: bool, n_samples: int, n_steps: int, output: str):
    input_dim, hidden_dim, output_dim = 2, 10, 2
    cell = RNNCell(input_dim, hidden_dim, output_dim)
    rnn = RNN(cell)

    key = random.PRNGKey(seed)

    rr.init(f"rnn_scratch_{type}", spawn=True)

    with open(f"{type}.json", "r") as f:
        data = json.load(f)

    trajectories = [jnp.array(trajectory).reshape(-1, 2, 1) for trajectory in data]
    points = jnp.concatenate(trajectories, axis=0)

    (min_x, min_y), (max_x, max_y) = jnp.min(points, axis=0), jnp.max(points, axis=0)
    jax.debug.print("min_x {min_x}, min_y {min_y}, max_x {max_x}, max_y {max_y}", min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y)

    shift = jnp.array([min_x, min_y])
    scale = jnp.array([max_x - min_x, max_y - min_y])
    for i in range(len(trajectories)):
        trajectories[i] = (trajectories[i] - shift) / scale

    for sample in range(n_samples):
        if use_input:
            x = jnp.array(trajectories[sample][:-1])
        else:
            x = jnp.array(trajectories[sample][:1])

        rr.set_time_sequence("step", 0)
        rr.log(f"points/{type}/x/{sample}", rr.Points2D(x))

        params = load_params(os.path.join(output, type, f"rnn_{type}_{epoch}.npz"))

        key, subkey = random.split(key)
        if use_input:
            if use_h0:
                h, y = rnn(params, x, init_carry=params[5])
            else:
                h, y = rnn(params, x, init_key=subkey)
        else:
            if use_h0:
                carry = params[5]
            else:
                carry = rnn.cell.initialize_carry(subkey)
            x = x[0]
            carries = [carry]
            ys = []
            for _ in range(n_steps):
                carry, y = rnn.cell(params, carry, x)
                carries.append(carry)
                ys.append(y)
                x = y
            h, y = jnp.array(carries), jnp.array(ys)

        for step in range(len(y)):
            rr.set_time_sequence("step", step)
            [rr.log(f"liens/{type}/h/{sample}/node{i}", rr.TimeSeriesScalar(h[step][i])) for i in range(hidden_dim)]  # type: ignore -- h[step][i] is a scalar
            rr.log(f"points/{type}/y/{sample}", rr.Points2D(y[: step + 1]))


@click.group()
def rnn_scratch():
    pass


rnn_scratch.add_command(train)
rnn_scratch.add_command(predict)
