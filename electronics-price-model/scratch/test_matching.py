import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import find_matching_laptops, safe_float

brand = "ASUS"
model = "Asus Vivobook"
ram = 4
storage = 256
storage_type = "SSD"
cpu = "i3"
generation = 13

est_price, match_score, top_matches = find_matching_laptops(
    brand, model, ram, storage, storage_type, cpu, generation
)

print("est_price:", est_price)
print("match_score:", match_score)
if top_matches is not None:
    print("top_matches count:", len(top_matches))
    print(top_matches[['Title', 'Price_Cleaned', 'match_score']].head(10).to_string())
