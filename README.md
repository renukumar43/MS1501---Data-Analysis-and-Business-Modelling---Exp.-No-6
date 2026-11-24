# Experiment 6 : Market Basket Analysis (Association Rule Mining using Apriori Algorithm)

## Aim
To perform Market Basket Analysis using the Apriori algorithm to identify frequent itemsets and generate association rules from transaction data for decision-making in retail and business analytics.

## Dataset
A sample retail transaction dataset containing **10 transactions** across **4 grocery items**: `Bread`, `Butter`, `Milk`, and `Eggs`.

| Transaction ID | Items Purchased |
| :---: | :--- |
| **T1** | Milk, Bread, Eggs |
| **T2** | Milk, Bread |
| **T3** | Milk, Eggs |
| **T4** | Bread, Butter |
| **T5** | Milk, Bread, Butter, Eggs |
| **T6** | Bread, Eggs |
| **T7** | Milk, Butter |
| **T8** | Milk, Bread, Butter |
| **T9** | Bread, Butter, Eggs |
| **T10** | Milk, Bread, Butter, Eggs |

*The transaction list is transformed into a One-Hot Encoded binary DataFrame prior to mining.*

## Algorithm / Steps Followed
1. Import required Python libraries (`pandas`, `mlxtend`).
2. Create the sample grocery transaction dataset.
3. Convert the dataset into a one-hot encoded binary matrix using `TransactionEncoder`.
4. Apply the `apriori()` algorithm (`min_support=0.3`) to identify frequent itemsets.
5. Generate association rules from frequent itemsets using `association_rules()` with `lift >= 1.0`.
6. Filter strong rules based on thresholds (`confidence >= 0.6` and `lift > 1`).
7. Interpret the rules to uncover cross-selling and store layout insights.

## Functions and Methods Used

| Category | Function / Method | Purpose / Description |
| :--- | :--- | :--- |
| **Data Handling** | `TransactionEncoder()` | Converts a list of transactions into a one-hot encoded binary matrix |
| **Frequent Pattern Mining** | `apriori()` | Finds frequent itemsets meeting the minimum support threshold |
| **Association Rule Generation** | `association_rules()` | Derives association rules using metrics such as support, confidence, and lift |
| **Filtering** | Conditional filtering (`rules[...]`) | Selects rules meeting strong confidence and lift thresholds |
| **Visualization / Display** | `print()` / `DataFrame.head()` | Displays encoded transactions, itemsets, and rules in tabular format |

## Program & Output

# 1. Import required Python libraries
import pandas as pd  
from mlxtend.preprocessing import TransactionEncoder  
from mlxtend.frequent_patterns import apriori, association_rules  

print("Libraries imported successfully.")  

# 2. Create the sample grocery transaction dataset
transactions = [
    ["Milk", "Bread", "Eggs"],
    ["Milk", "Bread"],
    ["Milk", "Eggs"],
    ["Bread", "Butter"],
    ["Milk", "Bread", "Butter", "Eggs"],
    ["Bread", "Eggs"],
    ["Milk", "Butter"],
    ["Milk", "Bread", "Butter"],
    ["Bread", "Butter", "Eggs"],
    ["Milk", "Bread", "Butter", "Eggs"]
]

print("Total transactions:", len(transactions))  
for transaction_id, items in enumerate(transactions, start=1):  
    print(f"T{transaction_id}: {', '.join(items)}")  

    # 3. Convert transactions into a one-hot encoded binary DataFrame
encoder = TransactionEncoder()  
encoded_array = encoder.fit(transactions).transform(transactions)  

basket = pd.DataFrame(encoded_array, columns=encoder.columns_)  

print("One-hot encoded transaction matrix:")  
display(basket.astype(int)) 

# 4. Apply Apriori to find frequent itemsets
frequent_itemsets = apriori(
    basket,
    min_support=0.3,
    use_colnames=True
)

frequent_itemsets = frequent_itemsets.sort_values(
    by=["support", "itemsets"],
    ascending=[False, True]
).reset_index(drop=True)  

print("Frequent itemsets (minimum support = 0.3):")  
display(frequent_itemsets)  

# 5. Generate association rules using lift as the metric
rules = association_rules(
    frequent_itemsets,
    metric="lift",
    min_threshold=1.0
)

print("Generated association rules:")  
if not rules.empty:
    display(rules[["antecedents", "consequents", "support", "confidence", "lift"]]
            .sort_values(by="lift", ascending=False).reset_index(drop=True))
else:
    print("No association rules were generated for the selected thresholds.")  

 # 6. Filter strong rules: confidence > 0.6 and lift > 1
strong_rules = rules[
    (rules["confidence"] > 0.6) &
    (rules["lift"] > 1)
].copy()

# Convert frozensets to readable strings for display
if not strong_rules.empty:
    strong_rules["antecedents"] = strong_rules["antecedents"].apply(
        lambda x: ", ".join(sorted(list(x)))
    )
    strong_rules["consequents"] = strong_rules["consequents"].apply(
        lambda x: ", ".join(sorted(list(x)))
    )
    strong_rules = strong_rules[
        ["antecedents", "consequents", "support", "confidence", "lift"]
    ].sort_values(by=["confidence", "lift"], ascending=False).reset_index(drop=True)

print("Strong association rules (confidence > 0.6 and lift > 1):")  
display(strong_rules)  

# 7. Display the most frequent itemsets and rule count
print("Number of frequent itemsets:", len(frequent_itemsets))
print("Number of generated association rules:", len(rules))  
print("Number of strong association rules:", len(strong_rules))  

print("\nFrequent itemsets with their support:") 
display(frequent_itemsets.assign(
    itemsets=frequent_itemsets["itemsets"].apply(lambda x: ", ".join(sorted(list(x))))  
))

print("\nStrong rules:")  
display(strong_rules)  

## Result
