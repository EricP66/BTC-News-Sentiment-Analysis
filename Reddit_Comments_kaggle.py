import os
import pandas as pd
import kagglehub

# Download Dataset from Kaggle (time between 2020.1 - 2022.04 )
path_1 = kagglehub.dataset_download("aglitoiumarius/rbitcoin-comments-202001202204",force_download=True)
# Download latest version
path_2 = kagglehub.dataset_download("jerryfanelli/reddit-comments-containing-bitcoin-2009-to-2019",force_download=True)


csv_files_1 =[i for i in  os.listdir(path_1) if i.endswith('csv')]
files_1 = os.path.join(path_1,csv_files_1[0])
df_1 = pd.read_csv(files_1)
print(f'Shape of Dataset between 2020.1 - 2022.04:{df_1.shape}')
print('-'*10)
print(df_1.columns.tolist())
print('-'*10)
print(df_1.head(3))

csv_files_2 =[i for i in  os.listdir(path_2) if i.endswith('csv')]
files_2 = os.path.join(path_2,csv_files_2[0])
df_2 = pd.read_csv(files_2)
print(f'Shape of Dataset between 2009.5 - 2019.12:{df_2.shape}')
print('-'*10)
print(df_2.columns.tolist())
print('-'*10)
print(df_2.head(3))

# transfer dataset_1 created_utc timestamp to datetime
df_1['datetime'] = pd.to_datetime(df_1['created_utc'],unit = 's',errors='coerce')

#only need time,text body and sentiment score and all in two datasets
columns = ['datetime','body','score']
available_columns = [col for col in columns if col in df_1.columns and col in df_2.columns]
df1 = df_1[available_columns]
df2 = df_2[available_columns]

#merge two datasets and order by time
df_total = pd.concat([df1,df2],axis=0, ignore_index=True)
print("Shape:", df_total.shape)

# 1. Datetime format
df_total['datetime'] = pd.to_datetime(df_total['datetime'], errors='coerce')

# 2. Datetime Analyze
print("\nData time analyze")
print(df_total['datetime'].describe())
print("Earlist:", df_total['datetime'].min())
print("Latest:", df_total['datetime'].max())
print("NaN:", df_total['datetime'].isna().sum())

# 3. Order by datetime
df_total = df_total.sort_values(
    by='datetime',
    ascending=True,
    na_position='last'   #Na Value will be in the end
).reset_index(drop=True)

# 4. Delete first duaplicates with same datetime and text
df_total_duplicates = df_total.drop_duplicates(subset=['datetime','body'],keep='first')
print(f'Delete {len(df_total)-len(df_total_duplicates)} duaplicates')
print(f'Total rows after Cleaning:{len(df_total_duplicates)}')

# df_total = df_total.dropna(subset=['datetime'])

print("\nBeginning 5 rows")
print(df_total_duplicates.head(5))

print("\nEnd 5 rows")
print(df_total_duplicates.tail(5))

# 4. Save the result
df_total_duplicates.to_parquet("bitcoin_comments_sorted.parquet", index=False)




