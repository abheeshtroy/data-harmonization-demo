# src/query.py

from src.utils       import load_env
from src.etl         import load_data, clean_data
from src.embeddings  import get_name_embeddings
from src.vector_db   import load_index, search_index

def main(
    csv_path: str = "data/AB_NYC_2019.csv",
    index_path: str = "name_index.faiss",
    k: int = 5
):
    # 1. Prepare environment & data
    _    = load_env()
    df   = clean_data(load_data(csv_path))
    
    # 2. Load FAISS index
    index = load_index(index_path)
    
    # 3. Pick a “probe” listing to query—here, the first one
    probe_name = df["name"].iloc[0]
    probe_emb  = get_name_embeddings([probe_name])[0]
    
    # 4. Search for top-k neighbors
    distances, indices = search_index(index, probe_emb, k=k)
    
    # 5. Display results
    print(f"\nProbe listing: {probe_name!r}\n")
    print(f"Top {k} similar listings:\n")
    for rank, (dist, idx) in enumerate(zip(distances, indices), 1):
        row = df.iloc[idx]
        print(f"{rank}. {row['name']!r}")
        print(f"   Borough: {row['neighbourhood_group']}, "
              f"Price: ${row['price']}, "
              f"Distance: {dist:.4f}\n")

if __name__ == "__main__":
    main()
