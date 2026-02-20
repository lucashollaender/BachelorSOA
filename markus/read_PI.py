import pandas as pd


def read_PI_values():

    # Read everything as string (important for Abaqus CSV)
    df = pd.read_csv("PI_values.csv", sep=",",
                     skipinitialspace=True, dtype=str)

    # Extract 13th column (index 12)
    U1_raw = df.iloc[:, 18].fillna("")

    # Remove all characters except valid float notation
    U1_raw = U1_raw.str.replace(r"[^0-9eE\+\-\.]", "", regex=True)

    # Convert to float
    U1 = pd.to_numeric(U1_raw, errors="coerce")

    # Keep zeros, drop NaN
    U1 = U1.dropna()

    print("U1 values (first 30):")
    print(U1.head(60))
    return


read_PI_values()
