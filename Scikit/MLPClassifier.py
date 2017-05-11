from sklearn.neural_network import MLPClassifier 

X = [[0., 0.], [1., 1.]]
Y = [0, 1]

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X, Y)

text_X = [[2., 2.], [-1., -2.]]
print(clf.predict(text_X))
print(clf.predict_proba(text_X) )