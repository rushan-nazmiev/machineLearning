import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report, f1_score


df = pd.read_csv('dataset_Facebook.csv',';')
df = df.drop('Type', axis=1)
#df = df[~df['Lifetime Post Consumers','Post Weekday','Paid','Lifetime Post Total Reach','Lifetime Post Total Impressions','Lifetime Engaged Users','Lifetime Post Consumptions','Lifetime Post Impressions by people who have liked your Page','Lifetime Post reach by people who like your Page','Lifetime People who have liked your Page and engaged with your post','comment','like','share'].isnull()]
df = df[~df['Post Weekday'].isnull()]
df = df[~df['Paid'].isnull()]
df = df[~df['Lifetime Post Total Reach'].isnull()]
df = df[~df['Lifetime Post Total Impressions'].isnull()]
df = df[~df['Lifetime Engaged Users'].isnull()]
df = df[~df['Lifetime Post Consumers'].isnull()]
df = df[~df['Lifetime People who have liked your Page and engaged with your post'].isnull()]
df = df[~df['Lifetime Post Consumptions'].isnull()]
df = df[~df['Lifetime Post Impressions by people who have liked your Page'].isnull()]
df = df[~df['Lifetime Post reach by people who like your Page'].isnull()]
df = df[~df['share'].isnull()]
df = df[~df['comment'].isnull()]

y = df['Paid']
df = df.drop('Paid', axis=1)
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.34, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
predict = model.predict(X_test)
sum_list = sum(abs(y_test - predict))
procent = accuracy_score(y_test, predict)

#print(sum_list)
#print(procent * 100)
print('Данный метод предсказывает {:.1%} верных ответов'.format(procent))

#print(df.info())

