import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np


data = pd.read_csv("export_dataframe.csv")
X = data[['acousticness', 'danceability', 'energy','instrumentalness',
            'liveness', 'loudness','speechiness','tempo', 'playlists']] #keep the playlists
Y = data[['playlists']]
sc = StandardScaler()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

#later we'll check our predictions on Test Set



#Standardising the training and test set
X_train_standard = sc.fit_transform(X_train[['acousticness', 'danceability', 'energy','instrumentalness',
            'liveness', 'loudness','speechiness','tempo']]) #remove playlists
X_test_standard = sc.transform(X_test[['acousticness', 'danceability', 'energy','instrumentalness',
            'liveness', 'loudness','speechiness','tempo']]) #remove playlists
            #transforms data on the same fit as on training data

sns.set()
#Plotting all of the data points against each other
# sns.pairplot(data[['acousticness', 'danceability', 'energy','instrumentalness',
#             'liveness', 'loudness','speechiness','tempo', 'playlists']],hue='playlists')



#PCA Fitting
model = PCA() #keep all the components
#keep only 2 components that maximise variance
model.fit(X_train_standard)
X_pca_train = model.transform(X_train_standard)
X_pca_test = model.transform(X_test_standard)




#Plotting training set
# plot_train = pd.DataFrame({'PCA1': X_pca_train[:, 0], 'PCA2': X_pca_train[:, 1]}) #first 2 PCA
X_train['PCA1'] = X_pca_train[:, 0]
X_train['PCA2'] = X_pca_train[:, 1]
figure = sns.lmplot('PCA1', 'PCA2', hue='playlists', data=X_train, fit_reg=False);
plt.show(figure)

model.explained_variance_ratio_
# array([0.52994336, 0.17144889, 0.09623449, 0.09087627, 0.05084408,
#        0.0307084 , 0.02092777, 0.00901675])
#PCA1 explains 0.53 of variance and so on





#Using random forest to make predictions
from sklearn.ensemble import RandomForestClassifier

cl = RandomForestClassifier()
cl.fit(X_pca_train, Y_train)

#predicting the test set
Y_pred = cl.predict(X_pca_test)
#which we can compare with the actual Y_test

pred = Y_pred.tolist()
test = [val for sublist in Y_test.values.tolist() for val in sublist]
result = pd.DataFrame({'Actual': test, 'Predicted': pred})
result.to_csv (r'PCApred.csv', header=True)




#predicting on a song not in the list

immsong = pd.read_csv("immigrant_song.csv")
immsong_standard = sc.transform(immsong[['acousticness', 'danceability', 'energy','instrumentalness','liveness', 'loudness','speechiness','tempo']])
immsong_pca = model.transform(immsong_standard)
immsong_pred = cl.predict(immsong_pca)

print(immsong_pred)

newsong = pd.read_csv("new_songs.csv")
new_standard = sc.transform(newsong[['acousticness', 'danceability', 'energy','instrumentalness','liveness', 'loudness','speechiness','tempo']])
new_pca = model.transform(new_standard)
new_pred = cl.predict(new_pca)

print(new_pred)
