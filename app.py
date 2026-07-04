import re
import sqlite3
from pathlib import Path
from typing import Optional
 
import joblib
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os
import numpy as np
 
from utils.lime import generate_lime_html
from utils.recommendations import generate_recommendations
 
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "signup.db"

app = FastAPI(title="Student Performance Prediction")


templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

MODEL_PATH = "Models/Ext_stack.sav"
SCALER_PATH = "Models/scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

FEATURES = [

    "Occupation",
    "Sex",
    "Age",
    "Residency_in_Dormitory",
    "Number_of_Family_Members",
    "Parent_Cohabitation_Status",
    "Mother_Education",
    "Father_Education",
    "Mother_Job_Category",
    "Father_Job_Category",
    "Criminal_Records",
    "Legal_Guardian",
    "Average_Sport_Activity_Per_Week",
    "Average_Studying_Per_Week",
    "Average_Grades_Previous_Semester",
    "Having_Scholarship",
    "Financial_Support_Parents",
    "Using_Extracurricular_Classes",
    "Having_Extracurricular_Activities",
    "History_Mental_Illness",
    "Willingness_to_Studying",
    "History_Physical_Illness",
    "Marital_Status_Relationship",
    "Relationship_Quality_Family",
    "Free_Time_After_Classes",
    "Communication_Quality_Classmates",
    "Alcohol_Consumption_Week",
    "Alcohol_Consumption_Weekend",
    "Number_of_Absences"

]

LABELS = {1: "Bad",2: "Average",3: "Good",4: "Very Good"}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html",{"request": request})

@app.post("/predict",response_class=HTMLResponse)
async def predict(request: Request):
    try:
        form = await request.form()

        values = []

        for feature in FEATURES:

            value = form.get(feature)

            if value is None:

                raise ValueError(
                    f"Missing value: {feature}"
                )

            values.append(
                float(value)
            )

        x = np.array(
            values,
            dtype=np.float32
        ).reshape(1, -1)


        # ======================
        # SCALE
        # ======================

        x_scaled = scaler.transform(
            x
        )


        # ======================
        # PREDICT
        # ======================

        pred = model.predict(
            x_scaled
        )[0]


        # ======================
        # CONFIDENCE
        # ======================

        confidence = None

        probabilities = None

        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = model.predict_proba(
                x_scaled
            )[0]

            confidence = round(

                np.max(
                    probabilities
                ) * 100,

                2

            )


        prediction = LABELS.get(
            int(pred),
            str(pred)
        )

        feature_values = {
            feature: values[index]
            for index, feature in enumerate(FEATURES)
        }

        lime_html, lime_error = generate_lime_html(
            x_scaled[0],
            model,
            scaler,
            FEATURES,
        )

        recommendation_data, recommendation_warning = generate_recommendations(
            feature_values,
            prediction,
            confidence if confidence is not None else 0,
        )

        return templates.TemplateResponse(

            "result.html",

            {

                "request": request,

                "prediction": prediction,

                "confidence": confidence,

                "probabilities": probabilities,

                "lime_html": lime_html,

                "lime_error": lime_error,

                "recommendations": recommendation_data.get("recommendations", []),

                "suggestions": recommendation_data.get("suggestions", []),

                "recommendation_source": recommendation_data.get("source"),

                "recommendation_model": recommendation_data.get("model"),

                "recommendation_warning": recommendation_warning,

            }

        )


    except Exception as e:

        return templates.TemplateResponse(

            "result.html",

            {

                "request": request,

                "prediction": "Prediction Failed",

                "confidence": 0,

                "error": str(e)

            }

        )
   


    
# Public routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/graphs",response_class=HTMLResponse)
async def graphs(request: Request):
    return templates.TemplateResponse("graphs.html",{"request": request})


@app.get("/home",response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html",{"request": request})

@app.get("/login",response_class=HTMLResponse)
async def login(request: Request):
    return templates.TemplateResponse("signin.html",{"request": request})

@app.get("/logon",response_class=HTMLResponse)
async def logon(request: Request):
    return templates.TemplateResponse("signup.html",{"request": request})


@app.get("/signup")
async def signup_get(request: Request):
    """Sign Up Page Form render."""
    return templates.TemplateResponse("signup.html", {"request": request})
 
@app.post("/signup")
async def signup_post(request: Request):
    """Processes User Registration Form."""
    form_data = await request.form()
    username = form_data.get('user', '').strip()
    name = form_data.get('name', '').strip()
    email = form_data.get('email', '').strip()
    number = form_data.get('mobile', '').strip()
    password = form_data.get('password', '').strip()
 
    # Form Validation regex rules
    username_pattern = r'^.{6,}$'
    name_pattern = r'^[A-Za-z ]{3,}$'
    email_pattern = r'^[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}$'
    mobile_pattern = r'^[6-9][0-9]{9}$'
    password_pattern = r'^(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{8,}$'
 
    if not re.match(username_pattern, username):
        return templates.TemplateResponse("signup.html", {"request": request, "message": "Username must be at least 6 characters."})
    if not re.match(name_pattern, name):
        return templates.TemplateResponse("signup.html", {"request": request, "message": "Full Name must be at least 3 letters, only letters and spaces allowed."})
    if not re.match(email_pattern, email):
        return templates.TemplateResponse("signup.html", {"request": request, "message": "Enter a valid email address."})
    if not re.match(mobile_pattern, number):
        return templates.TemplateResponse("signup.html", {"request": request, "message": "Mobile must start with 6-9 and be 10 digits."})
    if not re.match(password_pattern, password):
        return templates.TemplateResponse("signup.html", {"request": request, "message": "Password must be at least 8 characters, with an uppercase letter, a number, and a lowercase letter."})
 
    # Save to SQLite
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS info (user TEXT, name TEXT, email TEXT, mobile TEXT, password TEXT)")
   
    cur.execute("SELECT 1 FROM info WHERE user = ?", (username,))
    if cur.fetchone():
        con.close()
        return templates.TemplateResponse("signup.html", {"request": request, "message": "Username already exists. Please choose another."})
   
    cur.execute("INSERT INTO info (user, name, email, mobile, password) VALUES (?, ?, ?, ?, ?)", (username, name, email, number, password))
    con.commit()
    con.close()
   
    return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
 
 
@app.get("/signin")
async def signin_get(request: Request):
    """Sign In Page Form render."""
    return templates.TemplateResponse("signin.html", {"request": request})
 
@app.post("/signin")
async def signin_post(request: Request):
    """Processes User Sign In authentication."""
    form_data = await request.form()
    mail1 = form_data.get('user', '').strip()
    password1 = form_data.get('password', '').strip()
   
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS info (user TEXT, name TEXT, email TEXT, mobile TEXT, password TEXT)")
   
    # Query user, password, and email to store email in session
    cur.execute("SELECT user, password, email FROM info WHERE user = ? AND password = ?", (mail1, password1))
    data = cur.fetchone()
    con.close()
 
    # Admin bypass check
    is_admin = (mail1 == 'admin' and password1 == 'admin')
 
    if data is None and not is_admin:
        return templates.TemplateResponse("signin.html", {"request": request, "message": "Invalid username or password."})    
 
    # Store user and email in session state
    request.session["user"] = mail1
    if data:
        request.session["email"] = data[2]
    else:
        request.session["email"] = DEFAULT_RECIPIENT_EMAIL
 
    return templates.TemplateResponse("home.html", {"request": request})
 


@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/", status_code=303)
    response.delete_cookie(key="session_user")
    return response


 

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)