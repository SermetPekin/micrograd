
import pandas as pd

def iris_data():
    """
    Iris data

    Fisher, R. (1936). Iris [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C56C76.

    below we are using a popular github gist that includes data.
    scikit learn package datasets module also may be used.

    """
    def fnc(d: str):
        dict_ = {
            'Setosa': 0,
            'Versicolor': 1,
            'Virginica': 2,

        }
        return dict_.get(d, d)
    # Iris data
    url = "https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv"
    df = pd.read_csv(url)
    df['variety'] = df['variety'].apply(fnc)
    return df