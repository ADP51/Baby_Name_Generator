import pandas as pd
import numpy as np
from tensorflow_core.python.keras import Sequential
from tensorflow_core.python.keras.layers import SimpleRNN, TimeDistributed
from tensorflow_core.python.layers.core import Dense

# Get vocabulary of Names dataset
def get_vocabulary(names):
    # Define vocabulary to be set
    all_chars = set()

    # Add the start and end token to the vocabulary
    all_chars.add('\t')
    all_chars.add('\n')

    # Iterate for each name
    for name in names:

        # Iterate for each character of the name
        for c in name:

            if c not in all_chars:
                # If the character is not in vocabulary, add it
                all_chars.add(c)

    # Return the vocabulary
    return all_chars


def get_max_len(names):
    max_len = 0
    for name in names:
        if len(name) > max_len:
            max_len = len(name)
    return max_len


# read in csv into dataframe
names = pd.read_csv("data/baby-names-cleaned.csv")

print(names)

# Insert a tab in front of all the names
names['name'] = names.name.apply(lambda x : '\t' + x)

# Append a newline at the end of every name
names['target'] = names.name.apply(lambda x : x[1:len(x)] + '\n')

#get vocabulary
all_chars = get_vocabulary(names.name)

# Create the mapping of the vocabulary chars to integers
char_to_idx = { char : idx for idx, char in enumerate(sorted(all_chars)) }

# Create the mapping of the integers to vocabulary chars
idx_to_char = { idx : char for idx, char in enumerate(sorted(all_chars)) }

# Print the dictionaries
print(char_to_idx)
print(idx_to_char)

# get the longest name in dataset
max_len = get_max_len(names.name)

vocabulary = get_vocabulary(names.name)

# Initialize the input vector
input_data = np.zeros((len(names.name), max_len+1, len(vocabulary)), dtype='float32')

# Initialize the target vector
target_data = np.zeros((len(names.name), max_len+1, len(vocabulary)), dtype='float32')

# Iterate for each name in the dataset
for n_idx, name in enumerate(names):
    # Iterate over each character and convert it to a one-hot encoded vector
    for c_idx, char in enumerate(name):
        input_data[n_idx, c_idx, char_to_idx[char]] = 1

# Iterate for each name in the dataset
for n_idx, name in enumerate(names):
    # Iterate over each character and convert it to a one-hot encoded vector
    for c_idx, char in enumerate(name):
        target_data[n_idx, c_idx, char_to_idx[char]] = 1

# Create a Sequential model
model = Sequential()

# Add SimpleRNN layer of 50 units
model.add(SimpleRNN(50, input_shape=(max_len+1, len(vocabulary)), return_sequences=True))

# Add a TimeDistributed Dense layer of size same as the vocabulary
model.add(TimeDistributed(Dense(len(vocabulary), activation='softmax')))

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam")

# Print the model summary
model.summary()

# Fit the model for 5 epochs using a batch size of 128
model.fit(input_data, target_data, batch_size=128, epochs=5)

# Create a 3-D zero vector and initialize it with the start token
output_seq = np.zeros((1, max_len+1, len(vocabulary)))
output_seq[0, 0, char_to_idx['\t']] = 1

# Get the probabilities for the first character
probs = model.predict_proba(output_seq, verbose=0)[:, 1, :]

# Sample vocabulary to get first character
first_char = np.random.choice(sorted(list(vocabulary)), replace=False, p=probs.reshape(54))

# Print the character generated
print(first_char)











