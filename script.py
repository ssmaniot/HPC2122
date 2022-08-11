from scipy.cluster.hierarchy import dendrogram, linkage
from time import time
from matplotlib import pyplot as plt
import numpy as np
import csv
import sys

def main():
    with open(sys.argv[1]) as f:
        reader = csv.reader(f)
        header = next(reader, None)
        data = np.array([[float(i) for i in row] for row in reader])
    groups = data[:,0].astype('int')
    X = data[:,1:]
    begin = time()
    Z = linkage(X, 'single')
    elapsed = time() - begin
    print('Execution time: ', elapsed, 's', sep='')
    print(Z)
    fig = plt.figure(figsize=(25,10))
    dn = dendrogram(Z)
    plt.show()


if __name__ == '__main__':
    main()
