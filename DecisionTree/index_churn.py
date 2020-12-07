import pandas as pd #импорт библиотек, которые потребуются.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier #импортируем модель деревьев решений
from io import StringIO
from IPython.display import Image
from scipy.stats import randint as randint
from scipy.stats import uniform


# Для блакнота
#%matplotlib inline

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (18,12)

df_churn = pd.read_csv('churn.csv')


def preproc(df_init):  # объявляем функцию
    df_preproc = df_init.copy()

    df_preproc = df_preproc.drop(['State', 'Area Code', 'Phone'], axis=1)  # удаляем малоинформативные столбцы

    df_preproc.loc[:, ["Int'l Plan", 'VMail Plan']] = \
        df_preproc.loc[:, ["Int'l Plan", 'VMail Plan']].replace(
            {'no': 0, 'yes': 1})  # делаем замену занчений в указанных столбцах
    # на 0 и 1
    df_preproc.loc[:, 'Churn?'] = df_preproc.loc[:, 'Churn?'].replace(
        {'False.': 0,  # аналогично делаем замену и в столбце 'Churn?'
         'True.': 1})
    return df_preproc

df_preproc = df_churn.pipe(preproc) # обрабоатываем датафрейм при помощи функции, описанной выше.

X, y = df_preproc.iloc[:, :-1].values, df_preproc.iloc[:, -1].values # разделяем датафрейм на два множества:
# данные для обучения и ответы.

try:
    from sklearn.model_selection import validation_curve
except ImportError:
    from sklearn.learning_curve import validation_curve

try:
    from sklearn.model_selection import StratifiedKFold
except ImportError:
    from sklearn.cross_validation import StratifiedKFold

model = DecisionTreeClassifier(random_state=123) # инициализируем модель

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=132) #разбиваем наше множество на 5 частей и перемешиваем "shuffle=True"

train_scores, valid_scores = validation_curve(model, X, y,       #задаем параметры для валидационной кривой.
                                              'max_depth', range(1, 20), # будем исследовать глубину дерева в диапазоне от 1 до 20
                                              cv=cv, scoring='roc_auc')

train_score_mean = train_scores.mean(axis=1)
train_score_std = train_scores.std(axis=1)
valid_scores_mean = valid_scores.mean(axis=1)
valid_scores_std = valid_scores.std(axis=1)

plt.fill_between(range(1,20), train_score_mean-train_score_std, train_score_mean+train_score_std, color='b',
                 interpolate=True, alpha=0.5,)
plt.fill_between(range(1,20), valid_scores_mean-valid_scores_std, valid_scores_mean+valid_scores_std, color='r',
                 interpolate=True, alpha=0.5)

plt.plot(range(1,20), train_score_mean, c='b', lw=2)
plt.plot(range(1,20), valid_scores_mean, c='r', lw=2)

plt.xlabel('max depth')
plt.ylabel('ROC AUC')



try:
    from sklearn.model_selection import RandomizedSearchCV  # импортирует случайный поиск
except ImportError:
    from sklearn.cross_validation import RandomizedSearchCV

RND_SEED = 123

param_grid = {                                 #зададим параметры по которым и будем осуществлять поиск
    'criterion': ['gini', 'entropy'],
    'max_depth': randint(2, 8),
    'min_samples_leaf': randint(5, 10),
    'class_weight': [None, 'balanced']}

cv = StratifiedKFold(n_splits=5, random_state=123, shuffle=True)

model = DecisionTreeClassifier(random_state=123)
random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=200, n_jobs=-1,
                                   cv=cv, scoring='roc_auc', random_state=123)

random_search.fit(X, y)
best_model = random_search.best_estimator_;
best_model #параметры наилучшей модели

model = random_search.best_estimator_
imp = model.feature_importances_

pd.Series(index=df_preproc.columns[:-1], data = imp).sort_values()

print(random_search.best_params_)#выведем наилучшие параметры
print(random_search.best_score_)#лучшее значение метрики