import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

def main():
    # 修改数据路径 - 使用相对路径
    data_path = "winequality-red.csv"
    
    # 检查文件是否存在
    if not os.path.exists(data_path):
        print(f"错误：找不到数据文件 {data_path}")
        print(f"当前目录文件列表：{os.listdir('.')}")
        return
    
    # 读取数据
    data = pd.read_csv(data_path, sep=";")
    
    # 数据预处理
    X = data.drop("quality", axis=1)
    y = data["quality"]
    
    print("数据集信息：")
    print(f"数据形状：{data.shape}")
    print("列名：", data.columns.tolist())
    print("缺失值统计：")
    print(data.isnull().sum())
    print("\n质量分布：")
    print(y.value_counts().sort_index())

    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,
        random_state=42,
        stratify=y  # 保持类别分布
    )
    
    print(f"\n训练集样本数：{X_train.shape[0]}")
    print(f"测试集样本数：{X_test.shape[0]}")

    # 训练模型
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("模型训练完成！")

    # 预测和评估
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n模型准确率：{accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\n详细分类报告：")
    print(classification_report(y_test, y_pred))
    
    # 特征重要性
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n特征重要性排序：")
    print(feature_importance)

if __name__ == "__main__":
    main()
