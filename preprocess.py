import pandas as pd
import numpy as np
import re


def get_list_of_csi_data(file_path):

    csi_line_pattern = re.compile('.*CSI_DATA.*')
    csi_data_word_pattern = re.compile('CSI_DATA')

    final_data = []
    with open(file_path, 'r') as f:
        for line in f:
            match_obj = csi_line_pattern.match(line)  # find lines with CSI data
            if match_obj:
                # add comma between csi data and timestamp
                comma_added_before_csi_data = csi_data_word_pattern.sub(',CSI_DATA', match_obj.group())
                data_line = re.split(',', comma_added_before_csi_data)
                final_data.append(data_line)

    return final_data


def create_df_from_csi_list(csi_list):

    df = pd.DataFrame(csi_list)

    df = df.drop(0, axis=0)  # drop 1st row, which has column names

    df = df[[0, 4, 15, 24, 25, 26]]  # select required columns
    df.columns = ['abs_timestamp', 'rssi', 'noise_floor', 'rel_timestamp', 'csi_len', 'csi']  # rename required columns

    return df


def format_str_to_np_array_of_floats(string):

    list_of_strings = list(string.replace('[', '').replace(']', '').split())
    np_array_of_floats = np.array(list_of_strings, dtype=float)

    size = 128
    if len(np_array_of_floats) < size:
        np_array_of_floats = np.pad(np_array_of_floats, (0, size - np_array_of_floats.size))
    elif len(np_array_of_floats) > size:
        np_array_of_floats = np_array_of_floats[0:size]

    return np_array_of_floats


def get_amp_from_complex_num(np_array):

    odd_list = np_array[1::2]
    even_list = np_array[0::2]

    return np.sqrt(odd_list**2 + even_list**2)


def get_phase_from_complex_num(np_array):

    odd_list = np_array[1::2]
    even_list = np_array[0::2]

    return np.arctan(even_list/odd_list)  # imaginary comes before real in csi data


def add_csi_amp_and_phase(df):

    df.abs_timestamp = pd.to_datetime(df.abs_timestamp, unit='s', errors='coerce')
    df = df[df.abs_timestamp != pd.NaT]
    df.csi = df.csi.map(format_str_to_np_array_of_floats)

    df['csi_amp'] = df.csi.map(get_amp_from_complex_num)
    df['csi_phase'] = df.csi.map(get_phase_from_complex_num)

    df.rssi = df.rssi.astype(float)
    df.noise_floor = df.noise_floor.astype(float)

    return df


def get_csi_df_from_file(file_path):

    csi_list = get_list_of_csi_data(file_path)
    csi_df = create_df_from_csi_list(csi_list)
    csi_df = add_csi_amp_and_phase(csi_df)

    return csi_df


def get_filtered_csi_amp_and_phase_matrix(csi_df):
    csi_amp_matrix = np.array(csi_df.csi_amp.to_list())
    csi_phase_matrix = np.array(csi_df.csi_phase.to_list())

    csi_amp_matrix = np.concatenate((csi_amp_matrix[:, 6:30], csi_amp_matrix[:, 34:58]), axis=1)
    csi_phase_matrix = np.concatenate((csi_phase_matrix[:, 6:30], csi_phase_matrix[:, 34:58]), axis=1)
    csi_phase_matrix = csi_phase_matrix[1000:, :]

    max_amp = np.average(np.percentile(csi_amp_matrix, 99, axis=1))
    for csi_row in csi_amp_matrix:
        csi_row[csi_row > max_amp] = max_amp

    return csi_amp_matrix, csi_phase_matrix



if __name__ == "__main__":

    file_path = '../data/activity_detection/jumping_1.txt'
    df = get_csi_df_from_file(file_path)
    print('Shape of Dataframe: {}'.format(df.shape))
    print('Column names: {}'.format(df.columns))
    print(df.head())
