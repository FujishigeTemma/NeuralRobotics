import jax.numpy as jnp
import jax
import flax.linen as nn


class MTRNNCell(nn.RNNCellBase):
    @nn.compact
    def __call__(self, ):
        x = nn.Dense(2)(x)
        x = 

@click.command()
def train():
