import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt
import category_encoders as ce
from sklearn import preprocessing


columns_to_be_used_as_input = ["data_center", "orchestrator", "input_size_gb", "ram_gb", "processor", "available_disk_tb"]
column_to_be_used_as_output = ["duration_s"]

gawa = pd.read_csv("data/input/gawa.csv")
gawa = gawa[columns_to_be_used_as_input + column_to_be_used_as_output]

photoz = pd.read_csv("data/input/photoz.csv")
photoz = photoz[columns_to_be_used_as_input + column_to_be_used_as_output]

df = pd.concat([gawa, photoz])

# hot encoder
# encoder = ce.OneHotEncoder(use_cat_names=True)
# df_encoded = encoder.fit_transform(df)
# df_encoded.head()

# label encoder
le = preprocessing.LabelEncoder()
df['data_center'] = le.fit_transform(df['data_center'])
df['orchestrator'] = le.fit_transform(df['orchestrator'])
df['processor'] = le.fit_transform(df['processor'])


x = df[columns_to_be_used_as_input]
y = df[column_to_be_used_as_output]

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

logistic_regression= LogisticRegression()
logistic_regression.fit(X_train,y_train)
y_pred=logistic_regression.predict(X_test)

confusion_matrix = pd.crosstab(y_test["duration_s"], y_pred, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)

print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))
plt.show()

# Testing Model

# teste = {'POSCOMP': 65, 'Inglês': 6, 'Artigos publicados': 2}
# dft = pd.DataFrame(data = teste,index=[0])
# print(dft)
# resultado = logistic_regression.predict(dft)