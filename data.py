import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


data = pd.read_csv("/mnt/admin/pipeline/example/dataset/wine/winequality-red.csv", sep=";")
X = data.drop("quality", axis=1)
y = data["quality"]
print("修正后的列名：")
print(data.columns.tolist())
print("缺失值统计：")
print(data.isnull().sum())


X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    random_state=42
)
print("训练集样本数：{}".format(X_train.shape[0]))
print("测试集样本数：{}".format(X_test.shape[0]))
    

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("训练完成")
    

y_pred = model.predict(X_test)
print("准确率：{:.2f}".format(accuracy_score(y_test, y_pred)))
print("\n预测结果评估：")
print(classification_report(y_test, y_pred))
