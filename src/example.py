# https://github.com/google/lightweight_mmm/blob/main/examples/simple_end_to_end_demo.ipynb
import jax.numpy as jnp
import numpyro
from lightweight_mmm import lightweight_mmm
from lightweight_mmm import optimize_media
from lightweight_mmm import plot
from lightweight_mmm import preprocessing
from lightweight_mmm import utils

SEED = 105
data_size = 104 + 13
n_media_channels = 3
n_extra_features = 1

media_data, extra_features, target, costs = utils.simulate_dummy_data(
    data_size=data_size,
    n_media_channels=n_media_channels,
    n_extra_features=n_extra_features)

print("media_data")
print(media_data)
print(media_data.shape)

print("extra_features")
print(extra_features)
print(extra_features.shape)

print("target")
print(target)
print(target.shape)

print("costs")
print(costs)
print(costs.shape)

# Split and scale data.
split_point = data_size - 13
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

# correlations, variances, spend_fractions, variance_inflation_factors = preprocessing.check_data_quality(
#     media_data=media_scaler.transform(media_data),
#     target_data=target_scaler.transform(target),
#     cost_data=costs,
#     extra_features_data=extra_features_scaler.transform(extra_features))

# print(correlations)

mmm = lightweight_mmm.LightweightMMM(model_name="carryover")
number_warmup=1000
number_samples=1000

# mmm.fit(
#     media=media_data_train,
#     media_prior=costs,
#     target=target_train,
#     extra_features=extra_features_train,
#     number_warmup=number_warmup,
#     number_samples=number_samples,
#     seed=SEED)

# mmm.print_summary()

file_path = "media_mix_model.pkl"
# utils.save_model(media_mix_model=mmm, file_path=file_path)

loaded_mmm = utils.load_model(file_path=file_path)
loaded_mmm.trace["coef_media"].shape # Example of accessing any of the model values.
loaded_mmm.print_summary()

fig = plot.plot_media_channel_posteriors(media_mix_model=loaded_mmm)
fig.savefig('a.png')

fig = plot.plot_model_fit(loaded_mmm, target_scaler=target_scaler)
fig.savefig('b.png')

new_predictions = loaded_mmm.predict(media=media_scaler.transform(media_data_test),
                              extra_features=extra_features_scaler.transform(extra_features_test),
                              seed=SEED)
print(new_predictions.shape)
print(new_predictions)

fig = plot.plot_out_of_sample_model_fit(out_of_sample_predictions=new_predictions,
                                 out_of_sample_target=target_scaler.transform(target[split_point:]))
fig.savefig('c.png')