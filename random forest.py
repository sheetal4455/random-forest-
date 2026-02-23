from google.colab import files
uploaded = files.upload()
import pandas as pd
filename = list(uploaded.keys())[0]
df = pd.read_csv(filename)
print("Dataset Preview:")
print(df.head())
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Sex'] = df['Sex'].map({'male':0,'female':1})
X = df[['Pclass','Sex','Age','Fare']]
y = df['Survived']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = RandomForestClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test,y_pred)
print("Model Accuracy:",acc)
def predict_survival(pclass,sex,age,fare):
    sex_val = 1 if sex=="female" else 0
    data = [[pclass,sex_val,age,fare]]
    pred = model.predict(data)[0]
    result = "Passenger will Survive ✅" if pred==1 else "Passenger will Not Survive ❌"
    return f"{result}\nModel Accuracy: {acc:.2f}"
import gradio as gr
ui = gr.Interface(
    fn=predict_survival,
    inputs=[
        gr.Number(label="Passenger Class (1/2/3)"),
        gr.Dropdown(["male","female"],label="Gender"),
        gr.Number(label="Age"),
        gr.Number(label="Fare")
    ],
    outputs="text",
    title="Random Forest Titanic Survival Prediction"
)
ui.launch()
