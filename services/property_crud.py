from sqlalchemy.orm import Session
from typing import List, Optional
from database.models.property import Property
from database.schemas import PropertyCreate, PropertyUpdate
from fastapi import HTTPException


class PropertyCRUD:
    @staticmethod
    def create_property(db: Session, property_data: dict) -> Property:

        existing_property = (
            db.query(Property)
            .filter(
                (Property.property_name == property_data["property_name"])
                & (Property.location == property_data["location"])
            )
            .first()
        )

        if existing_property:
            raise HTTPException(
                status_code=400,
                detail="Property with this name and location already exists",
            )

        try:
            db_property = Property(**property_data)
            db.add(db_property)
            db.commit()
            db.refresh(db_property)
            return db_property
        except Exception as e:
            db.rollback()
            raise HTTPException(
                status_code=400, detail=f"Error creating property: {str(e)}"
            )

    @staticmethod
    def get_all_properties(db: Session) -> List[Property]:
        return db.query(Property).all()

    @staticmethod
    def get_property_by_id(db: Session, property_id: int) -> Property:
        property_obj = (
            db.query(Property).filter(Property.property_id == property_id).first()
        )
        if property_obj is None:
            raise HTTPException(status_code=404, detail="Property not found")
        return property_obj

    @staticmethod
    def get_properties_by_location(db: Session, location: str) -> List[Property]:
        properties = (
            db.query(Property).filter(Property.location.ilike(f"%{location}%")).all()
        )
        if not properties:
            raise HTTPException(
                status_code=404, detail="No properties found in this location"
            )
        return properties

    @staticmethod
    def get_properties_by_type(db: Session, property_type: str) -> List[Property]:
        properties = (
            db.query(Property)
            .filter(Property.property_type.ilike(f"%{property_type}%"))
            .all()
        )
        if not properties:
            raise HTTPException(
                status_code=404, detail="No properties found of this type"
            )
        return properties

    @staticmethod
    def get_available_properties(db: Session) -> List[Property]:
        return db.query(Property).filter(Property.availability == True).all()

    @staticmethod
    def update_property(db: Session, property_id: int, update_data: dict) -> Property:
        db_property = (
            db.query(Property).filter(Property.property_id == property_id).first()
        )
        if db_property is None:
            raise HTTPException(status_code=404, detail="Property not found")

        # Check for unique constraints if updating name and location
        if "property_name" in update_data and "location" in update_data:
            existing_property = (
                db.query(Property)
                .filter(
                    Property.property_id != property_id,
                    (Property.property_name == update_data.get("property_name"))
                    & (Property.location == update_data.get("location")),
                )
                .first()
            )

            if existing_property:
                raise HTTPException(
                    status_code=400,
                    detail="Another property with this name and location already exists",
                )

        try:
            for key, value in update_data.items():
                setattr(db_property, key, value)

            db.commit()
            db.refresh(db_property)
            return db_property
        except Exception as e:
            db.rollback()
            raise HTTPException(
                status_code=400, detail=f"Error updating property: {str(e)}"
            )

    @staticmethod
    def delete_property(db: Session, property_id: int) -> dict:
        db_property = (
            db.query(Property).filter(Property.property_id == property_id).first()
        )
        if db_property is None:
            raise HTTPException(status_code=404, detail="Property not found")

        try:
            db.delete(db_property)
            db.commit()
            return {"message": "Property deleted successfully"}
        except Exception as e:
            db.rollback()
            raise HTTPException(
                status_code=400, detail=f"Error deleting property: {str(e)}"
            )
