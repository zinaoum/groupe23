import json
import pandas as pd
import matplotlib.pyplot as plt

path = "dataset/annotations/instances_train2017.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

categories = {
    cat["id"]: cat["name"]
    for cat in data["categories"]
}

counts = {}

for ann in data["annotations"]:
    name = categories[ann["category_id"]]
    counts[name] = counts.get(name, 0) + 1

df = pd.DataFrame(
    list(counts.items()),
    columns=["Category", "Count"]
)

df = df.sort_values("Count", ascending=False)

print("\nObject Categories Statistics:")
print(df.to_string(index=False))

plt.figure(figsize=(14, 7))
plt.bar(df["Category"], df["Count"])
plt.xticks(rotation=90)
plt.title("Object Category Distribution - COCO Train2017")
plt.xlabel("Object Category")
plt.ylabel("Number of Objects")
plt.tight_layout()

plt.savefig("results/figures/category_distribution.png")
plt.show()