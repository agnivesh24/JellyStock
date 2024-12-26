from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from app import db
from datetime import datetime
import pandas as pd
from datetime import timedelta

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    name = db.Column(db.String(100))
    items = db.relationship('Item', backref='owner', lazy=True)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Item(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    category = db.Column(db.String(50))
    price = db.Column(db.Float, nullable=False)
    current_stock = db.Column(db.Integer, default=0)
    reorder_point = db.Column(db.Integer, default=0)
    lead_time = db.Column(db.Integer, default=1)  # in days
    supplier = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    sales_history = db.relationship('SalesHistory', backref='item', lazy=True)

class SalesHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    item_id = db.Column(db.Integer, db.ForeignKey('item.id'), nullable=False)
    date = db.Column(db.DateTime, nullable=False)
    units_sold = db.Column(db.Integer, nullable=False)
    price = db.Column(db.Float, nullable=False)
    stock_level = db.Column(db.Integer)
    discount = db.Column(db.Float, default=0)  # percentage
    lead_time = db.Column(db.Integer)  # in days
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'date': self.date,
            'units_sold': self.units_sold,
            'price': self.price,
            'stock': self.stock_level,
            'discount': self.discount,
            'lead_time': self.lead_time
        }

def get_historical_data(item_id, days=365):
    """
    Retrieve historical sales data for an item
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    sales_data = db.session.query(
        SalesHistory.date,
        SalesHistory.units_sold,
        SalesHistory.price,
        SalesHistory.stock_level,
        SalesHistory.discount,
        SalesHistory.lead_time
    ).filter(
        SalesHistory.item_id == item_id,
        SalesHistory.date.between(start_date, end_date)
    ).order_by(SalesHistory.date).all()
    
    return pd.DataFrame(sales_data, columns=[
        'date', 'units_sold', 'price', 'stock', 'discount', 'lead_time'
    ]) 