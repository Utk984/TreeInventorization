def batch_process(df, batch_size):
    """
    Splits a Pandas DataFrame into batches of rows.

    Args:
        df (pd.DataFrame): The DataFrame to process in batches.
        batch_size (int): The size of each batch.

    Yields:
        pd.DataFrame: A batch of rows from the DataFrame.
    """
    for i in range(0, len(df), batch_size):
        yield df.iloc[i:i + batch_size]
