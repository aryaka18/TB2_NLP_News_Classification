import json
import pandas as pd

# Load JSON data
with open('nlp/News_Category_Dataset_v3.json', 'r') as f:
    data = [json.loads(line) for line in f]

# Convert to a DataFrame
df = pd.DataFrame(data)
# print(df.head())  # Verify the data

# Select relevant columns (e.g., text, category)
df = df[['headline', 'short_description', 'category', 'date','link']]
# Save as CSV
df.to_csv('nlp/news.csv', index=False)

print("JSON file successfully converted to CSV!")

