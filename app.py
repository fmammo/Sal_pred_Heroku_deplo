from flask import Flask,jsonify, request
import pickle
import pandas as pd


app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello world"

@app.route('/predict/')
def salary_pred():
    model = pickle.load(open('model.pickle', 'rb'))
    Interview = request.args.get('Interview')
    Test = request.args.get('Test')
    
test_df = pd.DataFrame({'interview':[Interview], 'test':[Test]})
    
predict_salary = model.predict(test_df)
return jsonify({'Salary':str(predict_salary)})


if __name__=="__main__":
    app.run()
    
    
    
