import jax.numpy as jnp
from lightweight_mmm import plot
from lightweight_mmm import utils
from lightweight_mmm import preprocessing

import pandas as pd
import sys
sys.path.append('/src')

from src.testData import getMediaData

SEED = 105
data_size = 214

media_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
extra_features_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
target_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)


split_point = data_size - 30
df = pd.read_csv('/src/data/data1.csv')
media_data = getMediaData(df)
media_data_test = media_data[split_point:, ...]
revenueDf = pd.read_csv('/src/data/data3.csv')
extra_features = revenueDf['r1usd'].to_numpy().reshape(-1,1)
extra_features_test = extra_features[split_point:, ...]
target = revenueDf['r7usd'].to_numpy()

target_train = target[:split_point]
media_data_train = media_data[:split_point, ...]
extra_features_train = extra_features[:split_point, ...]

media_data_train = media_scaler.fit_transform(media_data_train)
extra_features_train = extra_features_scaler.fit_transform(extra_features_train)
target_train = target_scaler.fit_transform(target_train)

file_path = "test1.pkl"
loaded_mmm = utils.load_model(file_path=file_path)
loaded_mmm.trace["coef_media"].shape # Example of accessing any of the model values.
loaded_mmm.print_summary()
media_names = ['apple','applovin','bytedance','facebook','google','snapchat','unity']
fig = plot.plot_media_channel_posteriors(media_mix_model=loaded_mmm,channel_names=media_names)
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

fig = plot.plot_media_baseline_contribution_area_plot(media_mix_model=loaded_mmm,
                                                target_scaler=target_scaler,
)
                                                # fig_size=(30,10))
fig.savefig('d.png')