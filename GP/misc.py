import json

def read_json(path : str ):
    """
    Read the given json file.

    Parameters
    ----------
    path
     Read file on given path.

    Returns
    -------
    dict
    """
    try:
        with open(path) as file:
            data = json.load(file)
        return data
    except Exception as e:
        print(e)