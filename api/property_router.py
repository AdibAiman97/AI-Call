from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from database.connection import get_db
from services.property_crud import PropertyCRUD
from database.schemas import PropertyCreate, PropertyResponse, PropertyUpdate

router = APIRouter(prefix="/properties", tags=["properties"])


@router.post("/", response_model=PropertyResponse)
def create_property(property: PropertyCreate, db: Session = Depends(get_db)):
    return PropertyCRUD.create_property(db, property)


@router.get("/{property_id}", response_model=PropertyResponse)
def get_property(property_id: int, db: Session = Depends(get_db)):
    db_property = PropertyCRUD.get_property_by_id(db, property_id)
    if db_property is None:
        raise HTTPException(status_code=404, detail="Property not found")
    return db_property


@router.get("/", response_model=List[PropertyResponse])
def get_all_properties(db: Session = Depends(get_db)):
    return PropertyCRUD.get_all_properties(db)


@router.get("/available", response_model=List[PropertyResponse])
def get_available_properties(db: Session = Depends(get_db)):
    return PropertyCRUD.get_available_properties(db)


@router.get("/location/{location}", response_model=List[PropertyResponse])
def get_properties_by_location(location: str, db: Session = Depends(get_db)):
    return PropertyCRUD.get_properties_by_location(db, location)


@router.get("/type/{property_type}", response_model=List[PropertyResponse])
def get_properties_by_type(property_type: str, db: Session = Depends(get_db)):
    return PropertyCRUD.get_properties_by_type(db, property_type)


@router.put("/{property_id}", response_model=PropertyResponse)
def update_property(
    property_id: int, property: PropertyUpdate, db: Session = Depends(get_db)
):
    db_property = PropertyCRUD.update_property(db, property_id, property)
    if db_property is None:
        raise HTTPException(status_code=404, detail="Property not found")
    return db_property


@router.delete("/{property_id}", status_code=204)
def delete_property(property_id: int, db: Session = Depends(get_db)):
    if not PropertyCRUD.delete_property(db, property_id):
        raise HTTPException(status_code=404, detail="Property not found")
    return {"message": "Property deleted successfully"}
