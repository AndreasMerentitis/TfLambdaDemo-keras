import numpy as np
import pandas
import matplotlib.pyplot as plt

from keras import layers, optimizers, models
from sklearn.preprocessing import LabelEncoder

import tempfile
import urllib.request
train_file = tempfile.NamedTemporaryFile()
test_file = tempfile.NamedTemporaryFile()
urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", train_file.name)
urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", test_file.name)

import pandas as pd
COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]
df_train = pd.read_csv(train_file, names=COLUMNS, skipinitialspace=True)
df_test = pd.read_csv(test_file, names=COLUMNS, skipinitialspace=True, skiprows=1)


LABEL_COLUMN = "label"
df_train[LABEL_COLUMN] = (df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
df_test[LABEL_COLUMN] = (df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "gender", "native_country"]
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]

full_data = df_train
data = np.zeros(shape=(full_data.shape[0], full_data.shape[1]), dtype=np.float32)

transform_needed = [False,
           True,
           False,
           True,
           False,
           True,
           True,
           True,
           True,
           True,
           False,
           False,
           False,
           True,
           True]

for i in range(len(transform_needed)):
    if transform_needed[i]:
        tmp_data = full_data.iloc[:, i].tolist()
        encoder = LabelEncoder()
        encoder.fit(tmp_data)
        data[:, i] = encoder.transform(tmp_data)
    else:
        data[:, i] = full_data.iloc[:, i].tolist()

#print(pandas.unique(full_data.iloc[:, 14]))
train_size = int(len(data) * .8)

x_train = data[:train_size, :13]
y_train = data[:train_size, 14]

x_test = data[train_size:, :13]
y_test = data[train_size:, 14]

total_class = np.unique(data[:train_size, 14]).shape[0]

model = models.Sequential()
model.add(layers.Dense(16, input_shape=(x_train.shape[1],), activation="sigmoid"))
model.add(layers.Dropout(0.30))
model.add(layers.Dense(8, activation="sigmoid"))
model.add(layers.Dropout(0.30))
model.add(layers.Dense(1, activation="sigmoid"))

opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_split=0.33, epochs=20, batch_size=16)

loss, accuracy = model.evaluate(x_test, y_test)
print("Test Acc : " + str(accuracy))
print("Test Loss : " + str(loss))

# serialize model to JSON
model_json = model.to_json()
with open("model_ML.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save('model_ML.h5')

print("Saved model to disk")

plt.plot(history.history['acc'])
plt.plot(history.history['loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['acc', 'loss'], loc='upper left')
plt.show()
