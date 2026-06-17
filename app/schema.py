from pydantic import BaseModel, Field
from enum import Enum

class FuelType(str, Enum):
    PETROL = 'Petrol'
    DIESEL = 'Diesel'
    CNG = 'CNG'
    LPG = 'LPG'
    ELECTRIC = 'Electric'

class SellerType(str, Enum):
    INDIVIDUAL = 'Individual'
    DEALER = 'Dealer'
    
class TransmissionType(str, Enum):
    MANUAL = 'Manual'
    AUTOMATIC = 'Automatic'

class CarFeature(BaseModel):
    Car_Name: str = Field(default='Maruti 800', description='Car name')
    Company: str = Field(default='Maruti', description='Car company')
    Year: int = Field(default=2003, description='Car manufacturing year')
    Fuel_Type: FuelType = Field(default='Petrol', description='Fuel type')
    Seller_Type: SellerType = Field(default='Dealer', description='Seller type')
    Transmission: TransmissionType = Field(default='Manual', description='Transmission type')
    Owner: int = Field(default=0, description='Owner count')
    Kms_Driven: int = Field(default=100000, description='Kilometers driven')
    
    
class PredictionResponse(BaseModel):
    prediction: float = Field(..., description='Predicted price of the car')