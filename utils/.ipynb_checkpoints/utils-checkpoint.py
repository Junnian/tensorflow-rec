import pickle


def write_pickle(data,file):
    with open(file, "wb") as f:
        pickle.dump(data, f)
        
def read_pickle(file):
    with open(file, "rb") as f:
        data = pickle.load(f)
    return data