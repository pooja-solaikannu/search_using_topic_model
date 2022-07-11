import pandas as pd

removals = ["url", "Soil pH Preferences", "Inflorescence Height", "Plant Spread", "Awards and Recognitions", "Parentage",
            "Child plants", "Plant Height", "Minimum cold hardiness", "Toxicity", "Bloom Size"]

def basic_cleaning(x):
    x = x.replace("/", ",")
    x = x.replace("(", "")
    x = x.replace(")", "")
    
    return x

def data_cleaning(filepath):
    data = pd.read_excel(filepath, engine="openpyxl")
    data.fillna("", inplace=True)

    final_columns = list(set(data.columns) - set(removals))
    trimmed_data = data[final_columns]

    trimmed_data["combined_text"] = trimmed_data[final_columns].agg(lambda x: ' '.join(x.values), axis=1).T
    trimmed_data["cleaned_text"] = trimmed_data["combined_text"].apply(lambda x: basic_cleaning(x))
    trimmed_data["index"] = trimmed_data.index

    return trimmed_data
