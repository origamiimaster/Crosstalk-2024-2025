import pandas as pd

if __name__ == "__main__":
    compound_smiles_df = pd.read_csv('smiles.csv')
    print(compound_smiles_df)
    results_rescaled_as_df = pd.read_csv('rescaled_screen_results.txt', names=['score'])
    print(results_rescaled_as_df)

    merged_df = pd.concat([compound_smiles_df, results_rescaled_as_df], axis=1)
    print(merged_df)

    merged_df = merged_df.sort_values(by='score', ascending=False)
    print(merged_df)
    merged_df.to_csv("sorted_results.csv")

    top_100_df = merged_df[:100]
    print(top_100_df)
    top_100_df.to_csv("top100.csv")