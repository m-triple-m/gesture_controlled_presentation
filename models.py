from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()

class Presentation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationships
    slides = db.relationship('Slide', backref='presentation', lazy=True, cascade='all, delete-orphan')
    sessions = db.relationship('PresentationSession', backref='presentation', lazy=True, cascade='all, delete-orphan')

class Slide(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    presentation_id = db.Column(db.Integer, db.ForeignKey('presentation.id'), nullable=False)
    slide_number = db.Column(db.Integer, nullable=False)
    image_path = db.Column(db.String(500), nullable=False)
    
    # Relationships
    annotations = db.relationship('Annotation', backref='slide', lazy=True, cascade='all, delete-orphan')

class Annotation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    slide_id = db.Column(db.Integer, db.ForeignKey('slide.id'), nullable=False)
    annotation_data = db.Column(db.Text, nullable=False)  # JSON string of annotation points
    annotation_type = db.Column(db.String(50), default='drawing')

class PresentationSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    presentation_id = db.Column(db.Integer, db.ForeignKey('presentation.id'), nullable=False)
    session_name = db.Column(db.String(200))
    current_slide = db.Column(db.Integer, default=0)
    started_at = db.Column(db.DateTime, default=datetime.utcnow)
    ended_at = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)

class GestureLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('presentation_session.id'))
    gesture_type = db.Column(db.String(50), nullable=False)  # left, right, draw, erase, etc.
    gesture_data = db.Column(db.Text)  # JSON string of gesture details
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
