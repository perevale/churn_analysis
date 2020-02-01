import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
import statsmodels.api as stat
from sklearn import metrics, preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import pickle
import sklearn.externals
from yellowbrick.classifier import ClassificationReport


def read_data(source):
    source = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", source))
    df = pd.read_csv(source)  # , index_col=0
    df.drop(['CustomerID'], axis=1, inplace=True)
    return df


def visualize_ratio_churn(df):
    plt.rc("font", size=14)
    sns.set(style="white")
    sns.set(style="whitegrid", color_codes=True)

    # print(df["Churn"].value_counts())
    sns.countplot(x="Churn", data=df)
    plt.title("Ratio of Churn")
    plt.show()

    churned = len(df[df['Churn'] == "Yes"])
    unchurned = len(df[df['Churn'] == "No"])
    print(churned, " of users churned, while ", unchurned, "of users did not churn.")
    total = churned + unchurned
    churned_percentage = churned / total
    unchurned_percentage = unchurned / total
    print(churned_percentage * 100, "% of users churned, while ", unchurned_percentage * 100,
          "% of users did not churn.")


def most_profitable(df):
    d = df[df['Churn'] == 'Yes']
    # d.drop(['CustomerID'], axis=1, inplace=True)
    d = d.sort_values(by=['TotalCharges'], ascending=False)
    d = d.iloc[0:int(d['Churn'].count() * 0.1)]
    d = d.mode()
    print(d.iloc[0])
    print("Tenure mean: ", d['Tenure'].mean())
    print("Monthly Charges mean: ", d['MonthlyCharges'].mean())
    print("Total Charges mean:", d['TotalCharges'].mean())


def to_numeric(df):
    df['Gender'] = np.where(df['Gender'] == 'Male', 0, df['Gender'])
    df['Gender'] = np.where(df['Gender'] == 'Female', 1, df['Gender'])
    df['Gender'] = pd.to_numeric(df['Gender'])
    df['Partner'] = np.where(df['Partner'] == 'No', 0, df['Partner'])
    df['Partner'] = np.where(df['Partner'] == 'Yes', 1, df['Partner'])
    df['Partner'] = pd.to_numeric(df['Partner'])
    df['Dependents'] = np.where(df['Dependents'] == 'No', 0, df['Dependents'])
    df['Dependents'] = np.where(df['Dependents'] == 'Yes', 1, df['Dependents'])
    df['Dependents'] = pd.to_numeric(df['Dependents'])
    df['PhoneService'] = np.where(df['PhoneService'] == 'Yes', 1, df['PhoneService'])
    df['PhoneService'] = np.where(df['PhoneService'] == 'No', 0, df['PhoneService'])
    df['PhoneService'] = pd.to_numeric(df['PhoneService'])
    df['Churn'] = np.where(df['Churn'] == 'Yes', 1, df['Churn'])
    df['Churn'] = np.where(df['Churn'] == 'No', 0, df['Churn'])
    df['Churn'] = pd.to_numeric(df['Churn'])
    df['PaperlessBilling'] = np.where(df['PaperlessBilling'] == 'Yes', 1, df['PaperlessBilling'])
    df['PaperlessBilling'] = np.where(df['PaperlessBilling'] == 'No', 0, df['PaperlessBilling'])
    df['PaperlessBilling'] = pd.to_numeric(df['PaperlessBilling'])
    return df


def visualize_ratio(df):
    pd.crosstab(df.PaymentMethod, df.Churn).plot(kind='bar')
    plt.title('Churn Frequency for Payment Method')
    plt.ylabel('Counts')
    plt.savefig("freq_payment")
    plt.show()

    pd.crosstab(df.Contract, df.Churn).plot(kind='bar')
    plt.title('Churn Frequency for Contract')
    plt.ylabel('Counts')
    plt.savefig("freq_contract")
    plt.show()

    pd.crosstab(df.PaperlessBilling, df.Churn).plot(kind='bar')
    plt.title('Churn Frequency for Paperless Billing')
    plt.ylabel('Counts')
    plt.savefig("freq_billing")
    plt.show()

    # NOT INTRESTING
    pd.crosstab(df.StreamingMovies, df.Churn).plot(kind='bar')
    plt.title('Churn Frequency for Streaming Movies')
    plt.ylabel('Counts')
    plt.savefig("freq_movies")
    plt.show()

    # NOT INTRESTING
    pd.crosstab(df.StreamingTV, df.Churn).plot(kind='bar')
    plt.title('Churn Frequency for Streaming TV')
    plt.ylabel('Counts')
    plt.savefig("freq_tv")
    plt.show()

    pd.crosstab(df.TechSupport, df.Churn).plot(kind='bar')
    plt.title('Churn Frequency for Tech Support')
    plt.ylabel('Counts')
    plt.savefig("freq_techsupport")
    plt.show()

    pd.crosstab(df.DeviceProtection, df.Churn).plot(kind='bar')
    plt.title('Churn Frequency for Device Protection')
    plt.ylabel('Counts')
    plt.savefig("freq_deviceprotection")
    plt.show()

    pd.crosstab(df.OnlineBackup, df.Churn).plot(kind='bar')
    plt.title('Churn Frequency for Online Backup')
    plt.ylabel('Counts')
    plt.savefig("freq_backup")
    plt.show()

    pd.crosstab(df.OnlineSecurity, df.Churn).plot(kind='bar')
    plt.title('Churn Frequency for Online Security')
    plt.ylabel('Counts')
    plt.savefig("freq_security")
    plt.show()

    pd.crosstab(df.InternetService, df.Churn).plot(kind='bar')
    plt.title('Churn Frequency for Internet Service')
    plt.ylabel('Counts')
    plt.savefig("freq_internet")
    plt.show()

    # NOT INTERESTING
    pd.crosstab(df.MultipleLines, df.Churn).plot(kind='bar')
    plt.title('Churn Frequency for Multiple Lines')
    plt.ylabel('Counts')
    plt.savefig("freq_lines")
    plt.show()

    pd.crosstab(df.PhoneService, df.Churn).plot(kind='bar')
    plt.title('Churn Frequency for Phone Service')
    plt.ylabel('Counts')
    plt.savefig("freq_phone")
    plt.show()

    pd.crosstab(df.Tenure, df.Churn).plot(kind='bar')
    plt.title('Churn Frequency for tenure')
    plt.ylabel('Counts')
    plt.savefig("freq_tenure")
    plt.show()

    pd.crosstab(df.Dependents, df.Churn).plot(kind='bar')
    plt.title('Churn Frequency for Dependents')
    plt.ylabel('Counts')
    plt.savefig("freq_dependents")
    plt.show()

    pd.crosstab(df.Partner, df.Churn).plot(kind='bar')
    plt.title('Churn Frequency for Partner')
    plt.ylabel('Counts')
    plt.savefig("freq_partner")
    plt.show()

    pd.crosstab(df.SeniorCitizen, df.Churn).plot(kind='bar')
    plt.title('Churn Frequency for Senior Citizen')
    plt.ylabel('Counts')
    plt.savefig("freq_senior")
    plt.show()

    # NOT INTERESTING
    pd.crosstab(df.Gender, df.Churn).plot(kind='bar')
    plt.title('Churn Frequency for Gender')
    plt.ylabel('Counts')
    plt.savefig("freq_gender")
    plt.show()

    n = df.groupby("Churn")["MonthlyCharges"].mean().to_numpy()
    plt.bar(x=[0, 1], height=n, align='center', width=0.3)
    plt.xticks([0, 1], ["No", "Yes"])
    plt.ylabel('mean')
    plt.title('Churn amounts for Monthly Charges')
    plt.show()

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
    n = df.groupby("Churn")["TotalCharges"].mean().to_numpy()
    plt.bar(x=[0, 1], height=n, align='center', width=0.3)
    plt.xticks([0, 1], ["No", "Yes"])
    plt.ylabel('mean')
    plt.title('Churn amounts for Total Charges')
    plt.show()


def to_dummies(df):
    cat_vars = ['Gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineBackup',
                'InternetService', 'OnlineSecurity', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
    for var in cat_vars:
        cat_list = 'var' + '_' + var
        cat_list = pd.get_dummies(df[var], prefix=var)  # , drop_first=True
        data1 = df.join(cat_list)
        df = data1
    cat_vars = ['CustomerID', 'Gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
    data_vars = df.columns.values.tolist()
    to_keep = [i for i in data_vars if i not in cat_vars]

    data_final = df[to_keep]
    return data_final


def oversampling(X, y):
    sm = SMOTE(random_state=5)
    print(np.shape(X), np.shape(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    columns = X_train.columns
    sm_data_X, sm_data_y = sm.fit_sample(X_train, y_train)
    sm_data_X = pd.DataFrame(data=sm_data_X, columns=columns)
    sm_data_y = pd.DataFrame(data=sm_data_y, columns=['Churn'])
    # we can Check the numbers of our data
    print("length of oversampled data is ", len(sm_data_X))
    print("Number of unchurned in oversampled data", len(sm_data_y[sm_data_y['Churn'] == 0]))
    print("Number of churned", len(sm_data_y[sm_data_y['Churn'] == 1]))
    print("Proportion of unchurned data in oversampled data is ",
          len(sm_data_y[sm_data_y['Churn'] == 0]) / len(sm_data_X))
    print("Proportion of churn data in oversampled data is ",
          len(sm_data_y[sm_data_y['Churn'] == 1]) / len(sm_data_X))
    return sm_data_X, sm_data_y, X_test, y_test


def lr_model(df):
    df = to_numeric(df)
    data_final = to_dummies(df)

    X = data_final.loc[:, data_final.columns != 'Churn']
    y = data_final.loc[:, data_final.columns == 'Churn']

    sm_data_X, sm_data_y, X_test, y_test = oversampling(X, y)

    lr = LogisticRegression()
    c = X.columns
    rfe = RFE(lr, 20)
    rfe = rfe.fit(sm_data_X, sm_data_y.values.ravel())
    print(rfe.support_)
    print(rfe.ranking_)
    # print(np.where(rfe.support_,X,0))
    feat_cols = X.columns[rfe.support_ == True]
    print(feat_cols)
    # feat_cols = ['SeniorCitizen_1', 'Dependents_1', 'OnlineBackup_Yes', 'InternetService_Fiber optic',
    #              'InternetService_No', 'OnlineSecurity_Yes', 'TechSupport_Yes', 'StreamingTV_Yes', 'Contract_One year',
    #              'Contract_Two year', 'PaperlessBilling_1', 'PaymentMethod_Electronic check',
    #              'PaymentMethod_Mailed check']
    # feat_cols = ['SeniorCitizen_0', 'Partner_1', 'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No', 'OnlineSecurity_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 'OnlineBackup_No', 'OnlineBackup_No internet service', 'TechSupport_No internet service', 'TechSupport_Yes', 'StreamingTV_No', 'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year', 'PaperlessBilling_0', 'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check']
    feat_cols = ['SeniorCitizen_0', 'Partner_1', 'OnlineBackup_No', 'InternetService_Fiber optic',
                 'InternetService_DSL', 'OnlineSecurity_No', 'TechSupport_Yes', 'StreamingTV_No', 'Contract_One year',
                 'Contract_Two year', 'PaperlessBilling_0', 'PaymentMethod_Bank transfer (automatic)',
                 'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Mailed check', 'TotalCharges']
    feat_cols = ['Contract_Month-to-month', 'InternetService_Fiber optic', 'PaymentMethod_Electronic check',
                 'OnlineSecurity_No', 'TechSupport_No', 'PhoneService_0',
                 # 'MultipleLines_No phone service',
                 'PaperlessBilling_1', 'OnlineBackup_No',
                 # 'DeviceProtection_No',
                 'SeniorCitizen_1',
                 'StreamingTV_Yes',
                 'StreamingMovies_Yes',
                 'Dependents_0', 'MonthlyCharges', 'TotalCharges'
                 ]

    X = sm_data_X[feat_cols]
    y = sm_data_y['Churn']

    logit_model = stat.Logit(y, X)
    result = logit_model.fit()  # method='lbfgs'
    print(result.summary2())

    res = lr.fit(X, y)
    print(res)
    #
    # weights = pd.Series(lr.coef_[0],index=X.columns.values)
    # weights = weights.sort_values(ascending=False)
    # print(weights)

    churned = len(y_test[y_test['Churn'] == 1])
    unchurned = len(y_test[y_test['Churn'] == 0])
    print(len(y_test), 'churned:', churned, 'unchurned:', unchurned)

    X_test = X_test[feat_cols]
    y_pred = lr.predict(X_test)
    y_proba = lr.predict_proba(X_test)
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(lr.score(X_test, y_test)))

    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    print(confusion_matrix)

    print(classification_report(y_test, y_pred))

    logit_roc_auc = roc_auc_score(y_test, lr.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, lr.predict_proba(X_test)[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()
    sklearn.externals.joblib.dump(res, 'logreg.pkl')
    file_handler = open('features.obj', 'wb')
    pickle.dump(feat_cols, file_handler)

    X_test = X_test.to_numpy()
    print(np.shape(X_test))
    # X_test = X_test.reshape([-1,1])
    y_test = y_test.to_numpy()
    # Instantiate the classification model and visualizer
    visualizer = ClassificationReport(lr, classes=[0, 1])
    visualizer.fit(X, y)  # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)  # Evaluate the model on the test data
    g = visualizer.poof()  # Draw/show/poof the data

    plot_confusion_matrix(y_test, y_pred, ["No", "Yes"])


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = ["No", "Yes"]  # classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
    return ax


def linear_svm_model(df):
    df = to_numeric(df)
    le = preprocessing.LabelEncoder()
    df["MultipleLines"] = le.fit_transform(df["MultipleLines"])
    df["InternetService"] = le.fit_transform(df["InternetService"])
    df["OnlineSecurity"] = le.fit_transform(df["OnlineSecurity"])
    df["OnlineBackup"] = le.fit_transform(df["OnlineBackup"])
    df["DeviceProtection"] = le.fit_transform(df["DeviceProtection"])
    df["TechSupport"] = le.fit_transform(df["TechSupport"])
    df["StreamingTV"] = le.fit_transform(df["StreamingTV"])
    df["StreamingMovies"] = le.fit_transform(df["StreamingMovies"])
    df["Contract"] = le.fit_transform(df["Contract"])
    df["PaymentMethod"] = le.fit_transform(df["PaymentMethod"])

    cols = [col for col in df.columns if col not in ['Churn']]
    data = df[cols]
    target = df['Churn']
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.30, random_state=10)

    # create an object of type LinearSVC
    svc_model = LinearSVC(penalty='l1', dual=False)
    # train the algorithm on training data and predict using the testing data
    pred = svc_model.fit(data_train, target_train).predict(data_test)
    # print the accuracy score of the model
    print("LinearSVC accuracy : ", accuracy_score(target_test, pred, normalize=True))

    # Instantiate the classification model and visualizer
    visualizer = ClassificationReport(svc_model, classes=[0, 1])
    visualizer.fit(data_train, target_train)  # Fit the training data to the visualizer
    visualizer.score(data_test, target_test)  # Evaluate the model on the test data
    g = visualizer.poof()  # Draw/show/poof the data

    plot_confusion_matrix(target_test, pred, ["No", "Yes"])


def test_model(name, df):
    # Load the pickled model
    model = joblib.load(name)
    filehandler = open('features.obj', 'rb')
    features = pickle.load(filehandler)
    X = df[features]
    # Use the loaded model to make predictions
    pred = model.predict_proba(X)

    return pred[:, 1]


if __name__ == '__main__':
    df = read_data(source="churn.csv")

    """
    for training:
    """
    # lr_model(df)

    """
    for testing:
    # logistic regression model is saved in the file 'logreg.pkl'
    """
    df = to_numeric(df)
    df = to_dummies(df)
    pred = test_model('logreg.pkl', df)
    print(pred)

    # visualize_ratio_churn(df)
    # most_profitable(df)
    # visualize_ratio(df)

    # linear_svm_model(df)
