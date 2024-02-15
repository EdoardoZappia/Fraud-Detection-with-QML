# Data Exploration

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

file_path = "../dataset/dataset.csv"

# Read the dataset into a DataFrame using Pandas
try:
    df = pd.read_csv(file_path)
    print("Dataset imported successfully.")
except FileNotFoundError:
    print("File not found. Please check the file path.")
    exit()


# Histogram of transaction amounts
# bin_edges = range(0, 1040, 20)
# amount_fraud_0 = df[df['fraud'] == 0]['amount']
# amount_fraud_1 = df[df['fraud'] == 1]['amount']

# fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# axes[0].hist(amount_fraud_0, bins=bin_edges, color='blue', alpha=0.5)
# axes[0].hist(amount_fraud_1, bins=bin_edges, color= 'red', alpha=0.3)

# axes[0].set_title('Histogram of transaction amounts')
# axes[0].set_xlabel('Amount')
# axes[0].set_ylabel('Count')
# axes[0].set_xlim(0, 1040)
# axes[0].set_ylim(0, 10000)
# axes[0].set_xticks(range (0, 1040, 100))

# axes[1].hist(amount_fraud_0, bins=bin_edges, color='blue', alpha=0.5)
# axes[1].hist(amount_fraud_1, bins=bin_edges, color= 'red', alpha=0.3)
# axes[1].set_title('Histogram of transaction amounts')
# axes[1].set_xlabel('Amount')
# axes[1].set_ylabel('Count')
# axes[1].set_xlim(0, 1040)
# axes[1].set_ylim(0, 1000)
# axes[1].set_xticks(range (0, 1040, 100))

# Legend
# axes[0].legend(['Non-fraud', 'Fraud'], loc='upper right')
# axes[1].legend(['Non-fraud', 'Fraud'], loc='upper right')

# Adjust layout
#plt.tight_layout()

# Save the figure
#plt.savefig('Histogram of transaction amounts.png')

# Show the figure
#plt.show()


# Comments

# As we can see by the graph the fraudolent transactions are more frequent in the lower amounts, 
# while the non-fraudolent transactions are more frequent in the higher amounts. So we can see the
# fraudolent transactions as outliers of the amount feature.
# So we expect the amount of transactions to be an important feature to predict the response variable.



# Bar plot of fraudulent payments by age category

#fraud_count = df[df['fraud'] == 1].groupby(['age', 'gender']).size().unstack(fill_value=0)

# Plot the bar plot
# fraud_count.plot(kind='bar', stacked=False, position=1, width=0.6, color=['blue', 'orange', 'green'])
# plt.title('Count of Fraudulent Payments by Age Category and Gender')
# plt.xlabel('Age Category')
# plt.ylabel('Count of Fraudulent Payments')
# plt.xticks(rotation=45)
# plt.legend(title='Gender')
# plt.tight_layout()

# plt.savefig('Count of Fraudulent Payments by Age Category and Gender.png')

#plt.show()


# Comment

# We can see that the most affecting age categories to fraudolent transactions are 26-35, 36-45 and 46-55.
# We can expect that result considering that those age categories are the most probable to do transactions.
# Females are constitute more fraudolent transactions.

# Since it's not clear, we will do further analysis to understand the relationship between gender, age and fraudolent transactions.



# Bar plot of fraudulent payments by merchant category

#df_fraudulent = df[df['fraud'] == 1]

# Group by merchant category and count the occurrences
# merchant_count = df_fraudulent.groupby('category').size().sort_values(ascending=False)

# plt.figure()

# Plot the bar plot
# merchant_count.plot(kind='bar')
# plt.title('Barplot of Fraudulent Payments by Merchant Category')
# plt.xlabel('Merchant Category')
# plt.ylabel('Count')
# plt.xticks(rotation=90)
# plt.tight_layout()

#plt.savefig('Barplot of Fraudulent Payments by Merchant Category.png')

#plt.show()


# Comment

# The distribution of fraudolent payments by merchant category points out that the most affected categories 
# are "sports and toys" and "health". This result emphasizes the importance of the merchant category 
# feature for our problem.





# PREPROCESSING

# 'step' represent the day of the simulation, so it's not a feature that we can use to predict the response variable.
df.drop(columns=['step'], inplace=True)
print(df.head())


num_categories = len(np.unique(df['customer']))
print('Unique customers: ',num_categories)

num_categories = len(np.unique(df['zipcodeOri']))
print('Unique zipcodeOri: ',num_categories)
df.drop(columns=['zipcodeOri'], inplace=True)
# 'zipcodeOri' has only one value so we drop it

num_categories = len(np.unique(df['merchant']))
print('Unique merchants: ',num_categories)

num_categories = len(np.unique(df['zipMerchant']))
print('Unique zipMerchant: ',num_categories)
df.drop(columns=['zipMerchant'], inplace=True)
# 'zipMerchant' has only one value so we frop it

num_categories = len(np.unique(df['category']))
print('Unique categories: ',num_categories)

print(df.head())

# The number of unique values for the 'customer' is 4112 so if we use
# one hot encoding we will have 4112 new columns. This is not feasible.

# Encoding of categorical variables using LabelEncoder

encoder = LabelEncoder()

encoded_customer = encoder.fit_transform(df['customer'])
df['customer'] = encoded_customer
encoded_age = encoder.fit_transform(df['age'])
df['age'] = encoded_age
encoded_gender = encoder.fit_transform(df['gender'])
df['gender'] = encoded_gender
encoded_merchant = encoder.fit_transform(df['merchant'])
df['merchant'] = encoded_merchant
encoded_category = encoder.fit_transform(df['category'])
df['category'] = encoded_category

print(df.head())


