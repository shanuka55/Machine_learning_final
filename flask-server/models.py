from sqlalchemy import Boolean, Column, Integer, String, DECIMAL
from database import Base

class Place(Base):
    __tablename__ = 'places'
    
    id = Column(Integer, primary_key=True, index=True)
    price = Column(String(50))
    qty = Column(String(50))
    contact = Column(String(50))
    discription = Column(String(50))
    latitude = Column(DECIMAL(9, 6))
    longitude = Column(DECIMAL(9, 6))
    
    