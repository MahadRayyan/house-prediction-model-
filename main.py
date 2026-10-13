import io # help python to work with the data in memory
from fastapi import FastAPI , HTTPException , UploadFile,File
from fastapi.responses import StreamingResponse  # use to send back response to the user
from pydantic import BaseModel , Field # used for validation 
import joblib # use to store your machine learning model
import pandas as pd

app=FastAPI()

model=joblib.load("house_model.joblib")
features=joblib.load("house_feature.joblib")

class HouseFeature(BaseModel):
    MedInc : float = Field(gt=0,description="Median income of neighbourhood")
    HouseAge : float =Field(gt=0,description="average House Age")
    AveRooms : float =Field(gt=0,description="average number rooms")
    AveBedrms : float =Field(gt=0,description="average number of bed rooms")
    Population : float =Field(gt=0,description="Total population in neighbourhood")
    AveOccup : float =Field(gt=0,description="average number of occupancy")
    Latitude : float =Field(gt=32,le=42,description="Latitude")
    Longitude : float =Field(gt=-125,le=114,description="Longitude")


@app.get("/")
def home():
    return {
        "message":"California housing api",
        "status":"Running",
        "endpoint":"send post request to /predict"
    
    }
@app.get("/health")
def health():
    return {
        "status": "running",
        "model": "RandomforestRegressor" ,
        "features": features ,
        "avg_error":"$32,773"
    }
@app.post("/predict")
def predict(house : HouseFeature):
    try:
        input_data=pd.DataFrame([{
                'MedInc':house.MedInc,
                'HouseAge':house.HouseAge,
                'AveRooms':house.AveRooms,
                'AveBedrms':house.AveBedrms,
                'Population':house.Population,
                'AveOccup':house.AveOccup,
                'Latitude':house.Latitude,
                'Longitude':house.Longitude
        }])

        predicted=model.predict(input_data)[0]
        price_usd=predicted*100000
        return {
            "predicted price":f"${price_usd:,.0f}",
            "predicted_price_short":f"${predicted:,.2f} hundred thousand",
            "confidence_range":f"${price_usd-32773} to ${price_usd+32773}"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"prediction failed:{str(e)}",
        )


@app.post("/predict_file")
async def predict_file(file:UploadFile=File(...)):  # three dots mean file is required


    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="please upload csv file only"

        )
    contents=await file.read()
    df=pd.read_csv(io.BytesIO(contents))
    required_columns =[
                        'MedInc',
                        'HouseAge',
                        'AveRooms',
                        'AveBedrms',
                        'Population',
                        'AveOccup',
                        'Latitude',
                        'Longitude'
    ]
    missing_columns=[
        col for col in required_columns
        if col not in df.columns
        
    ]
    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail=f"these columns are missing from your file {missing_columns}"
        )
    if len(df)==0:
        raise HTTPException(
            status_code=400,
            detail="upload file has no rows"
        )
    try:
        predictions=model.predict(df[required_columns])
        df["predict_columns_usd"] = predictions * 100000
        df["predict_columns_usd"]=df["predict_columns_usd"].apply(lambda x :f"${x:,.0f}")
        output = df.to_csv(index=False)
        return StreamingResponse(
            io.StringIO(output),
            media_type="text/csv",
            headers={
                "Content-Disposition" : "attachment; filename =prediction.csv"
            }
            
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"prediction failed: {str(e)}"
        )
    
