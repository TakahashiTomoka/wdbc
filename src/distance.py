import csv
from statistics import mean


def distance_mean(file_A, file_B):
    with open(file_A) as file:
        reader = csv.reader(file)
        mat_A = [row for row in reader]
    mat_A = [[float(v) for v in row] for row in mat_A]
    
    with open(file_B) as file:
        reader = csv.reader(file)
        mat_B = [row for row in reader]
    mat_B = [[float(v) for v in row] for row in mat_B]
    
    dist_len = []
    
    
    for i in range(len(mat_A)):
        sum_2 = 0
        for j in range(3,len(mat_A[i])):
            sum_2 += (mat_A[i][j] - mat_B[i][j])**2
        dist_len.append(sum_2**(1/(len(mat_A[i])-2)))
    print(mean(dist_len))
    
    