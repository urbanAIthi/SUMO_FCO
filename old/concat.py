import pickle
import os
import pandas as pd

def load_dataframes_from_pickle(filename):
    dataframes = list()
    with open(filename, 'rb') as file:
        while True:
            try:
                df = pickle.load(file)
                dataframes.append(df)
            except EOFError:
                break
    return dataframes



if __name__ == "__main__":
    filename = 'test.pkl'
    file_size = os.path.getsize(filename)
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    df_list = load_dataframes_from_pickle(filename)

    # set the chunk size
    chunk_size = 10000

    # create an empty list to hold the chunk DataFrames
    chunk_df_list = []

    # loop through the DataFrames in chunks
    for i in range(0, len(df_list), chunk_size):
        # get a chunk of DataFrames
        chunk = df_list[i:i + chunk_size]

        # concatenate the chunk DataFrames
        chunk_df = pd.concat(chunk)

        # add the chunk DataFrame to the list
        chunk_df_list.append(chunk_df)

    # concatenate the chunk DataFrames into a single DataFrame
    combined_df = pd.concat(chunk_df_list)