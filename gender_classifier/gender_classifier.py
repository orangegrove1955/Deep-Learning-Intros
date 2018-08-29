from sklearn import tree

# MTW 2018-8-29

# Data values
# [height,width,shoe size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

# Data labels
Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

# Create Decision Tree
clf = tree.DecisionTreeClassifier()

# Fit the data to decision tree
clf = clf.fit(X, Y)

# Create a prediction
prediction = clf.predict([[190, 70, 43]])

# Display prediction
print(prediction)