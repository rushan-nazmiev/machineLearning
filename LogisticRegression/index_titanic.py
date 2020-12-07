import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('34_titanic_train.csv')
df = df.drop(['Name','Sex', 'Ticket','Cabin','Embarked'],axis=1)
df = df[~df['Age'].isnull()]
y = df['Survived']
df = df.drop('Survived', axis=1)

# Обучение без разбивки на обучающую и тестовую выборки.
model_0 = LogisticRegression()
model_0.fit(df,y)
predict_0 = model_0.predict(df)
sum_list = sum(abs(y - predict_0))

precision = precision_score(y, predict_0)
recall = recall_score(y, predict_0)
accuracy = accuracy_score(y, predict_0)

print(sum_list)
print(classification_report(y, predict_0))

# Обучение с разбивкой на обучающую и тестовую выборки.
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.33, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
predict = model.predict(X_test)
sum_list = sum(abs(y_test - predict))

print(sum_list)
print(classification_report(y_test, predict))