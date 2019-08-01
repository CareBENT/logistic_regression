from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd


def logistic_regression():
    """
    逻辑回归做二分类进行癌症预测（根据细胞的属性特征）
    :return:
    """
    # 1、导入数据
    #    --当数据没有特征名称(列名)时，在导入时需要创建列明，否则会自动以第一行的内容作为列名
    column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                    'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                    'Normal Nucleoli', 'Mitoses', 'Class']
    data = pd.read_csv("./data/breast-cancer-wisconsin.data", names=column_names)

    # 2、缺失值处理
    #    --将‘？’用NaN代替
    data_process = data.replace(to_replace='?', value=np.nan, inplace=False)
    #    --删除含有NaN的数据
    data_process = data_process.dropna()

    # 3、数据集划分
    x_train, x_test, y_train, y_test = train_test_split(data_process.iloc[:, 1: 10],
                                                        data_process.iloc[:, 10],
                                                        test_size=0.25)

    # 4、特征工程，数据标准化
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)

    # 5、构造并训练模型
    lr = LogisticRegression(penalty='l2', C=1.0)
    lr.fit(x_train, y_train)

    predict = lr.predict(x_test)
    report = classification_report(y_test, predict, labels=[2, 4], target_names=['良性', '恶性'])
    print(report)
    coef = lr.coef_
    print(np.dot(coef, coef.T))

    return None


if __name__ == "__main__":
    logistic_regression()
