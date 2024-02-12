# Data Exploration

import pandas as pd
import matplotlib.pyplot as plt

file_path = "../DataExp/dataset.csv"

# Read the dataset into a DataFrame using Pandas
try:
    df = pd.read_csv(file_path)
    print("Dataset imported successfully.")
except FileNotFoundError:
    print("File not found. Please check the file path.")
    exit()


# Histogram of transaction amounts
bin_edges = range(0, 1040, 20)
amount_fraud_0 = df[df['fraud'] == 0]['amount']
amount_fraud_1 = df[df['fraud'] == 1]['amount']

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot histograms with intervals for each 'fraud' category
axes[0].hist(amount_fraud_0, bins=bin_edges, color='blue', alpha=0.5)
axes[0].hist(amount_fraud_1, bins=bin_edges, color= 'red', alpha=0.3)

axes[0].set_title('Histogram of transaction amounts')
axes[0].set_xlabel('Amount')
axes[0].set_ylabel('Count')
axes[0].set_xlim(0, 1040)
axes[0].set_ylim(0, 10000)
axes[0].set_xticks(range (0, 1040, 100))

axes[1].hist(amount_fraud_0, bins=bin_edges, color='blue', alpha=0.5)
axes[1].hist(amount_fraud_1, bins=bin_edges, color= 'red', alpha=0.3)
axes[1].set_title('Histogram of transaction amounts')
axes[1].set_xlabel('Amount')
axes[1].set_ylabel('Count')
axes[1].set_xlim(0, 1040)
axes[1].set_ylim(0, 1000)
axes[1].set_xticks(range (0, 1040, 100))

# Legend
axes[0].legend(['Non-fraud', 'Fraud'], loc='upper right')
axes[1].legend(['Non-fraud', 'Fraud'], loc='upper right')

# Adjust layout
plt.tight_layout()

# Save the figure
#plt.savefig('Histogram of transaction amounts.png')

# Show the figure
#plt.show()



# Bar plot of fraudulent payments by age category

fraud_count = df[df['fraud'] == 1].groupby(['age', 'gender']).size().unstack(fill_value=0)

# Plot the bar plot
fraud_count.plot(kind='bar', stacked=False, position=1, width=0.6, color=['blue', 'orange', 'green'])
plt.title('Count of Fraudulent Payments by Age Category and Gender')
plt.xlabel('Age Category')
plt.ylabel('Count of Fraudulent Payments')
plt.xticks(rotation=45)
plt.legend(title='Gender')
plt.tight_layout()

plt.savefig('Count of Fraudulent Payments by Age Category and Gender.png')

#plt.show()


# Bar plot of fraudulent payments by merchant category

df_fraudulent = df[df['fraud'] == 1]

# Group by merchant category and count the occurrences
merchant_count = df_fraudulent.groupby('category').size()

plt.figure()

# Plot the bar plot
merchant_count.plot(kind='bar')
plt.title('Barplot of Fraudulent Payments by Merchant Category')
plt.xlabel('Merchant Category')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.tight_layout()

#plt.savefig('Barplot of Fraudulent Payments by Merchant Category.png')

#plt.show()

