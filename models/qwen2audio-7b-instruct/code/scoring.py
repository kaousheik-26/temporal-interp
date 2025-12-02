import json
import glob

files = glob.glob("../results/*.json")

for f in files:
    correct = 0
    total = 0
    with open(f) as F:
        data = json.load(F)
    for item in data:
        correct += 1 if item["correct"] else 0

    print(f"{f.split('/')[-1]}: {correct}/{len(data)} = {correct/len(data):.4f}")

