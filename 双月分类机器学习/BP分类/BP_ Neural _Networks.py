import pandas as pd
raw = pd.read_csv('data.txt')
raw_data = raw.values
raw_feature = raw_data[0:22, 0:2]
label_feature=raw_data[0:22, 2:3]

test_raw=raw_data[22:, 0:2]
test_label= raw_data[22:, 2:3]


print(raw_feature[2][1])
