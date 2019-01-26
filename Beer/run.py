import pandas as pd
import numpy as np
from read import get_data
from read import get_dataset





if __name__ == '__main__':
    get_dataset('beer-ratings/train.csv','beer-ratings/test.csv')