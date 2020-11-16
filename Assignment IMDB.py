import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#Open dataset
cleanDB = pd.read_csv("cleanDB.csv")

#print the column names
def printCol(daFr):
    for col_name in daFr: 
        print(col_name)
        
#change column name
cleanDB.rename(columns={"Gross": "Gross ($M)"}, inplace=True)

#sum of null values in a column or check na values
db["Time Duration (min)"].isna().sum()

#number of unique values
db["Title"].nunique()
db['Title'].value_counts()      

#Save to CSV
cleanDB.to_csv('cleanDB.csv')
scaleDB.to_csv('scaleDB.csv')

#drop duplicates
test = db
top25 = top25.drop_duplicates(subset=['Movie Title'], keep='first')

#Used to copy column to new series. Then change Series to DF
db = cleanDB.copy()
db = db.to_frame(name="Director")

#Drop na values
db = db.dropna()

#Split columns and drop old column
split = split.to_frame(name="Cast")
db[['Director','Stars']] = db.Director.str.split("|",expand=True,)
Title["Movie Year"] = Title["Movie Year"].str.replace(r'\D', '').astype(int)

#Rename column
shc.rename(columns={"0": "Silhoutte"}, inplace=True)

#Drop column
cleanDB = cleanDB.drop(columns=["Unnamed: 0"])
db = cleanDB.drop(columns=["Unnamed: 0"])

#Sort Value
top25Rate = cleanDB.sort_values(by="Rate", ascending=False).head(25)

#Strip or replace
cleanDB["Director"] = cleanDB["Director"].str.strip()
cleanDB["Vote"] = cleanDB["Vote"].str.replace("\,", "")

#Change column to int
cleanDB["Gross ($M)"].astype(float).astype(int)

#Join dataframes
shc = shc.join(h)
top25 = pd.concat([top25Gross,top25Rate]).drop_duplicates().reset_index(drop=True)

#Pie
pieValue = cleanDB.Certificate.value_counts()
pieValue.plot.pie(autopct='%1.1f%%',figsize=(10,10), labels=None, wedgeprops = {'linewidth': 3} )
plt.ylabel("")
plt.legend(pieValue.index, loc=2, fontsize=10)
plt.title('Movie Certifications Counts')

#Fill na
cleanDB["Gross ($M)"] = cleanDB["Gross ($M)"].fillna(0)

#Reorder columns
cleanDB = cleanDB[["Movie Title", "Movie Year", "Certificate", "Duration", "Genre", "Rate", "Metascore", "Description", "123", "Stars", "Vote", "Gross ($M)", "Director"]]

#First genre
cleanDB['First Genre'] = cleanDB.Genre.apply(lambda x: x.split(',')[0] if ',' in x else x)

#Scatter plot
plt.scatter(cleanDB["Movie Year"], cleanDB["Gross ($M)"], marker="o")

#Simple line plot
cleanDB.plot(x="Movie Year", y="Gross ($M)")

cleanDB["Gross ($M)"].sum().plot()


#bar
db['Rate'].value_counts().sort_values(ascending=True).plot.barh()

db['Score'].plot.density()by=["Movie Revenue"]

seperate_genre='Action','Adventure','Animation','Biography','Comedy','Crime','Drama','Fantasy','Family','History','Horror','Music','Musical','Mystery','Romance','Sci-Fi','Sport','Thriller','War','Western'
for genre in seperate_genre:
    genreDB = cleanDB['Genre'].str.contains(genre).fillna(False)
    print('The total number of movies with ',genre,'=',len(cleanDB[genreDB]))
    f, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='Genre', data=cleanDB[genreDB], palette="Greens_d");
    plt.title(genre)
    compare_movies_rating = ['Runtime_Minutes', 'Votes','Revenue_Millions', 'Metascore']
    for compare in compare_movies_rating:
        sns.jointplot(x='Rating', y=compare, data=cleanDB[genreDB], alpha=0.7, color='b', size=8)
        plt.title(genre)




#Scatter graph
color = np.random.randint(0, 10, len(cleanDB["Gross ($M)"]))
plt.colorbar()
topG = plt.scatter(top25Gross["Rate"], top25Gross["Gross ($M)"], c='r', marker = 'o')
topR = plt.scatter(top25Rate["Rate"], top25Rate["Gross ($M)"], c='g', marker = 'x')
plt.grid(True)
plt.legend((topG, topR), ("Top 25 Grossing", "Top 25 Rated"), loc = 'upper right')
plt.xlabel('Rate')
plt.ylabel('Gross ($M)')
plt.title('Top 25 Movies for Both Rate and Gross')

#Simple line graph
top25Gross.groupby(["Rate"])["Gross ($M)"].mean().plot()
top25Rate.groupby(["Rate"])["Gross ($M)"].mean().plot()
plt.xlabel('Movie Rating')
plt.ylabel('Gross ($M)')


#Simple Bar
cleanDB["First Genre"].value_counts(dropna=True).sort_index()
.plot(kind='barh',figsize=(10,10),c='b')
plt.xlabel('Number of Movies')
plt.ylabel('Genre')
plt.title('Number of Movies of First Genre')

cleanDB.Director.value_counts()[:12]
.plot(kind='barh')
plt.xlabel('Number of Movies')
plt.ylabel('Directors')
plt.title('Top Directors')



cleanDB = cleanDB.sort_index(axis = 0,ascending=True)
top25Gross["Rate"].mean()

#Box plot
plt.figure(figsize=(14,10))
sns.boxplot(x='Rate',y='First Genre',data=cleanDB)


#Heatmap
corr = cleanDB.corr()
plt.figure(figsize=(13,10))
sns.heatmap(corr, cmap="YlGnBu", annot=True)


#KMeans
from sklearn import cluster
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import scale
from sklearn import cluster
from sklearn.preprocessing import LabelEncoder
import sklearn.metrics as metrics
from sklearn import preprocessing


scaleDB = cleanDB.copy()
scaleDB = scaleDB.dropna()
X = scaleDB.values[:, (2,4,6,7,11,12)]
Y = scaleDB.values[:, 0]

scaled_cleanDB = scale(X)
s1= []
s2= []
s3= []
n_samples, n_features = scaled_cleanDB.shape
n_digits = len(np.unique(Y))
Y2 = LabelEncoder().fit_transform(Y)

#Number of clusters
for k in range (2,50): 
    kmeans = cluster.KMeans(n_clusters=k)
    kmeans.fit(scaled_cleanDB)
    print(k)
    print("silhouette_score = ", metrics.silhouette_score(scaled_cleanDB,kmeans.labels_))
    s1.append(metrics.silhouette_score(scaled_cleanDB,kmeans.labels_))
    print("completeness_score = ", metrics.completeness_score(Y2,kmeans.labels_))
    s2.append(metrics.completeness_score(Y2,kmeans.labels_))
    print("homogeneity_score = ", metrics.homogeneity_score(Y2,kmeans.labels_))
    s3.append(metrics.homogeneity_score(Y2,kmeans.labels_))


shc = pd.DataFrame((s1,s2,s3), index = ["Silhouette", "Completeness", "Homogeneity"])
shc = shc.T

#Line graph for K-Means
shc.plot()
plt.legend(loc="upper right")
plt.xlabel('Number of Clusters')
plt.ylabel('Score')


#Logistic Regression
from sklearn import metrics
from sklearn import model_selection
from sklearn.preprocessing import scale
from sklearn import preprocessing
from sklearn import utils

dbData = db.values[:, (0,2,4,7,11, 12)]
#Target is the movie rates
dbTarget = db.values[:,6]

#Need to use this to change target values for int otherwise float error
lab_enc = preprocessing.LabelEncoder()
encoded = lab_enc.fit_transform(dbTarget)

dbData = scale(dbData)
dbTarget = scale(dbTarget)

dbData=dbData.astype(int)

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(dbData, encoded, test_size = 0.30)

from sklearn.linear_model import LogisticRegression
lm = LogisticRegression(max_iter=4000)
lm.fit(X_train, Y_train)
lm.predict_proba(X_test)
predicted = lm.predict(X_test)
print(metrics.classification_report(Y_test, predicted))
print(metrics.confusion_matrix(Y_test, predicted))





