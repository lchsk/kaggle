import pandas as pd


def save_submission(columns=None, data=None, output='my_output.csv'):
    if columns is None:
        raise ValueError("columns must be set")
    if data is None:
        raise ValueError("data must be set")

    results = {
        column: results
        for column, results in zip(columns, data)
    }

    data = pd.DataFrame(
        data=results,
        columns=columns,
    )
    data.to_csv(output, index=False)