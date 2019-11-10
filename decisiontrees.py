import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

data = pd.read_csv("export_dataframe.csv")

#Split the data into features and categories
feature_cols = ['acousticness', 'danceability', 'energy','instrumentalness','liveness', 'loudness','speechiness','tempo']
X = data[['acousticness', 'danceability', 'energy','instrumentalness',
            'liveness', 'loudness','speechiness','tempo']]
Y = data[['playlists']]

#split data into training and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

#Fitting using Decision Trees
cl = DecisionTreeClassifier()
cl.fit(X_train,Y_train)
Y_pred = cl.predict(X_test)

pred = Y_pred.tolist()
test = [val for sublist in Y_test.values.tolist() for val in sublist]
result = pd.DataFrame({'Actual': test, 'Predicted': pred})
result.to_csv (r'DecisionTrees.csv', header=True)

ac = metrics.accuracy_score(Y_test, Y_pred) #accuracy score of the programme
print(ac)
#0.9827586206896551

newsong = pd.read_csv("new_songs.csv")
new_pred = cl.predict(newsong[feature_cols])
print(new_pred)


from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus

dot_data = StringIO()
export_graphviz(cl, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_png('music3.png')
Image(graph.create_png())
