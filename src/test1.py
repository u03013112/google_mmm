import jax.numpy as jnp
import numpyro
from lightweight_mmm import lightweight_mmm
from lightweight_mmm import optimize_media
from lightweight_mmm import plot
from lightweight_mmm import preprocessing
from lightweight_mmm import utils

import pandas as pd
import sys
sys.path.append('/src')

from src.testData import getMediaData,getMediaCost

SEED = 105
data_size = 214
n_media_channels = 7
n_extra_features = 1

df = pd.read_csv('/src/data/data1.csv')
media_data = getMediaData(df)
costs = getMediaCost(df)

revenueDf = pd.read_csv('/src/data/data3.csv')
extra_features = revenueDf['r1usd'].to_numpy().reshape(-1,1)
target = revenueDf['r7usd'].to_numpy()

print(media_data.shape)
print(extra_features.shape)
print(target.shape)
print(costs.shape)

# Split and scale data.
split_point = data_size - 30
# Media data
media_data_train = media_data[:split_point, ...]
media_data_test = media_data[split_point:, ...]
# Extra features
extra_features_train = extra_features[:split_point, ...]
extra_features_test = extra_features[split_point:, ...]
# Target
target_train = target[:split_point]
     
media_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
extra_features_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
target_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
cost_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean, multiply_by=0.15)

media_data_train = media_scaler.fit_transform(media_data_train)
extra_features_train = extra_features_scaler.fit_transform(extra_features_train)
target_train = target_scaler.fit_transform(target_train)
costs = cost_scaler.fit_transform(costs)

mmm = lightweight_mmm.LightweightMMM(model_name="carryover")
number_warmup=1000
number_samples=1000

mmm.fit(
    media=media_data_train,
    media_prior=costs,
    target=target_train,
    # extra_features=extra_features_train,
    number_warmup=number_warmup,
    number_samples=number_samples,
    seed=SEED)

# mmm.print_summary()

file_path = "test2.pkl"
utils.save_model(media_mix_model=mmm, file_path=file_path)
