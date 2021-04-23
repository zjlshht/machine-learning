from pandas.io.parsers import read_csv
import numpy as np
import pandas as pd


def main():
    data = read_csv("data.csv")
    df = data.values  # 转换数据类型
    a1 = df[:, 0]
    a2 = df[:, 2]
    a3 = df[:, 3]
    a4 = df[:, 4]
    a5 = df[:, 5]  # 0-1
    a6 = df[:, 7]
    a7 = df[:, 8]
    a8 = df[:, 9]
    a9 = df[:, 10]
    a10 = df[:, 11]  # 0-1
    a11 = df[:, 13]  # pupolarity
    a12 = df[:, 15]
    a13 = df[:, 16]
    a14 = df[:, 17]
    a15 = df[:, 18]  # year
    pop_min = []
    pop_max = []
    for i in range(1920, 2022):
        pop_min.append(data[data['year'] == i]['popularity'].min())
        pop_max.append(data[data['year'] == i]['popularity'].max())
    for i in range(len(a15)):
        min_value = pop_min[a15[i]-1920]
        max_value = pop_max[a15[i]-1920]
        a11[i] = (a11[i]-min_value)/(max_value-min_value)
    X = np.array([a1, a2, a3, a4, a6, a7, a8, a9, a12, a13,
                 a14, a5, a10, a11]).T  # create new dataset
    final = pd.DataFrame(X)
    final.to_csv('Data_on_year.csv', float_format='%.2f', na_rep="NAN!")


if __name__ == "__main__":
    main()
