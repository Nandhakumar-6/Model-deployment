
# Https request: Hypertext Transfer Protocol Secure (HTTPS) is a protocol that secures communication and data transfer between a user's web browser and a website. 
# The most commonly used HTTP methods are:
# GET = The GET method is used to retrieve data on a server...
# POST = The POST method is used to create new resources by giving some information. ...
# PUT =  The PUT method is used to replace an existing resource with an updated version. ...
# PATCH = The PATCH method is used to update an existing resource. ...
# DELETE = delete the resources...


from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)   #starting point....

#1, load the trained model
with open('linear_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)  #loading the pickle file
    
#Considering Flask cap as a map which consists of different spots. So each spots is considered as each route(i.e., each location)
# for eg: 1st route will be starting spot(i.e., Home page)
#         2nd route will be answer submission spot like prediction spot..
#         3rd ...
#         4th ... etc
#         so, each different route is considered as each different page(spot)

# In map, Inbetween each spots it consists some special symbol which represents a spot to identify easily...and it is called as Decorators
#flask(map) ---> Each routes ---> Each decorators...>Each functions

@app.route('/')  # '/'- slash decorator which represents home page
#every route conduct's some function which is called as HTTPS actions...
def home():
    return render_template("index.html")  #render_template--->telling them to bring index.html when we use home page...


@app.route('/predict', methods = ['post'])   #'/predict' - slash predict decorator which represents prediction spot(predict button)....
#this prediction route has post request method...
def predict():
    #Get input data from the form
    mid_sem_marks = request.form['MSE']  #whatever writing in the MSE form field is getting and saved here....
    Attendance = request.form['Attendance']
    
    #Make a prediction using the loaded model
    input_data = [[float(mid_sem_marks)],[float(Attendance)]]  #converting input data(mid_sem_marks, Attendance) into float
    
    reshapped_data = np.array(input_data).reshape(-1, 1)  #reshapping the input data inorder to avoid shape problem
    prediction = model.predict(reshapped_data)  #prediction happen
    
    #pass the prediction value to the template
    return render_template('index.html', prediction= prediction[0]) #rendering and passing prediction value to display because of Ginga template....
    # while prediction there may be chance of getting array of values, so being that use index[0] to avoid any problem...
    
if __name__ == '__main__':
    app.run(debug=True)   
# to run this app, we have to make '__main__' to start and run the app. So, app.run(debug = True) 
#By doing this,the app will be started...

