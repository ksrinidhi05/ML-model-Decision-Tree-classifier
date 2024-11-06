import pandas as pd
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        target_column = request.form['target_column']
        data = pd.read_csv(file)

        data.replace('?', pd.NA, inplace=True)
        data.dropna(inplace=True)

        if target_column not in data.columns:
            return "Error: Target column not found in dataset."

        le = LabelEncoder()
        data[target_column] = le.fit_transform(data[target_column])
        class_names = le.inverse_transform(range(len(le.classes_)))

        X = data.drop(target_column, axis=1)
        y = data[target_column]

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        clf = DecisionTreeClassifier(criterion='gini', random_state=0)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        conf_matrix = confusion_matrix(y_test, y_pred)

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("Confusion Matrix:\n", conf_matrix)

        # Generate decision tree plot
        plt.figure(figsize=(20, 10))
        plot_tree(clf, filled=True, feature_names=data.columns[:-1], class_names=class_names.astype(str))  
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()  
        plt.close()  

        # Generate confusion matrix heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix Heatmap')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        cm_img = io.BytesIO()
        plt.savefig(cm_img, format='png')
        cm_img.seek(0)
        cm_plot_url = base64.b64encode(cm_img.getvalue()).decode()
        plt.close()

        # Get the head of the dataset
        head_data = data.head().to_html(classes='dataframe', index=False)

        return render_template('result.html', 
                               accuracy=accuracy, 
                               precision=precision, 
                               recall=recall, 
                               f1=f1, 
                               plot_url=plot_url, 
                               conf_matrix=conf_matrix.tolist(),
                               cm_plot_url=cm_plot_url,
                               head_data=head_data)  # Pass head data to template

    return render_template('upload.html')

@app.route('/upload-another', methods=['POST'])
def upload_another_file():
    return upload_file()

if __name__ == '__main__':
    plt.switch_backend('Agg')
    app.run(debug=True)
