from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression

models = {"logreg":LogisticRegression(),
            "svc":SVC(),
            "rf":RandomForestClassifier()}