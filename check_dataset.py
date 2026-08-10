import pandas as pd

#This script is used to check the dataset for any missing values or inconsistencies.
df = pd.read_csv('datasets/description_categories_dataset.csv')

print("*******************DATASET CHECKING******************")

# df = df[df["category"] != "Bills & Utilities"]


print("\ndataset shape: ")
print(df.shape) 

print("\ndataset head: ")
print(df.head())

print("null values in the dataset: ")
print(df.isnull().sum())

print("\nduplicate values in the dataset: ")
print(df.duplicated().sum())

print("\n category distribution in the dataset: ")
print(df['category'].value_counts())




# duplicates = df[df.duplicated()]
# print(duplicates.head(20))

# df = df.drop_duplicates()

# df.to_csv('datasets/description_categories_dataset.csv', index=False)