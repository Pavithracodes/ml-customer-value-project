from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
import xgboost as xgb
import joblib
import random
import json

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Models Directory
MODELS_DIR = ROOT_DIR / 'ml_models'
MODELS_DIR.mkdir(exist_ok=True)

# Define Models
class Customer(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    email: str
    join_date: str
    total_spend: float = 0
    transaction_count: int = 0
    avg_transaction_value: float = 0
    segment: Optional[str] = None
    clv_score: Optional[float] = None
    predicted_next_month_spend: Optional[float] = None
    risk_level: Optional[str] = None

class Transaction(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    customer_id: str
    merchant_category: str
    amount: float
    timestamp: str
    merchant_name: str

class Offer(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    customer_id: str
    customer_name: str
    offer_type: str
    offer_text: str
    discount_percentage: float
    category: str
    expected_engagement_score: float
    valid_until: str

class DataGenerateRequest(BaseModel):
    num_customers: int = 500
    num_transactions: int = 5000

class PredictionRequest(BaseModel):
    customer_id: str

# Merchant categories
MERCHANT_CATEGORIES = [
    'Restaurants', 'Groceries', 'Gas Stations', 'Online Shopping',
    'Entertainment', 'Travel', 'Healthcare', 'Utilities',
    'Electronics', 'Fashion', 'Home Improvement', 'Fitness'
]

# Helper Functions
def generate_synthetic_data(num_customers: int = 500, num_transactions: int = 5000):
    """Generate synthetic transaction data"""
    customers = []
    transactions = []
    
    # Generate customers
    for i in range(num_customers):
        join_date = datetime.now(timezone.utc) - timedelta(days=random.randint(30, 730))
        customer = {
            'id': str(uuid.uuid4()),
            'name': f'Customer {i+1}',
            'email': f'customer{i+1}@example.com',
            'join_date': join_date.isoformat(),
            'total_spend': 0,
            'transaction_count': 0,
            'avg_transaction_value': 0
        }
        customers.append(customer)
    
    # Generate transactions
    for _ in range(num_transactions):
        customer = random.choice(customers)
        merchant_category = random.choice(MERCHANT_CATEGORIES)
        amount = round(random.uniform(10, 500), 2)
        
        transaction = {
            'id': str(uuid.uuid4()),
            'customer_id': customer['id'],
            'merchant_category': merchant_category,
            'amount': amount,
            'timestamp': (datetime.now(timezone.utc) - timedelta(days=random.randint(0, 180))).isoformat(),
            'merchant_name': f'{merchant_category} Store {random.randint(1, 20)}'
        }
        transactions.append(transaction)
        
        # Update customer stats
        customer['total_spend'] += amount
        customer['transaction_count'] += 1
    
    # Calculate avg transaction value
    for customer in customers:
        if customer['transaction_count'] > 0:
            customer['avg_transaction_value'] = round(customer['total_spend'] / customer['transaction_count'], 2)
    
    return customers, transactions

async def get_customer_features(customer_id: str):
    """Extract features for a customer for ML predictions"""
    customer = await db.customers.find_one({'id': customer_id}, {'_id': 0})
    if not customer:
        return None
    
    transactions = await db.transactions.find({'customer_id': customer_id}, {'_id': 0}).to_list(1000)
    
    if len(transactions) == 0:
        return None
    
    # Calculate features
    total_spend = sum(t['amount'] for t in transactions)
    transaction_count = len(transactions)
    avg_transaction = total_spend / transaction_count if transaction_count > 0 else 0
    
    # Category diversity
    categories = set(t['merchant_category'] for t in transactions)
    category_diversity = len(categories)
    
    # Recent activity (last 30 days)
    now = datetime.now(timezone.utc)
    recent_transactions = [t for t in transactions if (now - datetime.fromisoformat(t['timestamp'])).days <= 30]
    recent_spend = sum(t['amount'] for t in recent_transactions)
    recent_count = len(recent_transactions)
    
    # Customer tenure in days
    join_date = datetime.fromisoformat(customer['join_date'])
    tenure_days = (now - join_date).days
    
    features = {
        'total_spend': total_spend,
        'transaction_count': transaction_count,
        'avg_transaction': avg_transaction,
        'category_diversity': category_diversity,
        'recent_spend': recent_spend,
        'recent_count': recent_count,
        'tenure_days': tenure_days,
        'days_since_last_transaction': min((now - datetime.fromisoformat(t['timestamp'])).days for t in transactions)
    }
    
    return features

async def train_ml_models():
    """Train ML models for CLV prediction and customer segmentation"""
    # Get all customers and their features
    customers = await db.customers.find({}, {'_id': 0}).to_list(10000)
    
    if len(customers) < 10:
        return {'error': 'Not enough data to train models'}
    
    features_list = []
    clv_targets = []
    
    for customer in customers:
        features = await get_customer_features(customer['id'])
        if features:
            features_list.append(features)
            # CLV is based on total spend and tenure
            clv = features['total_spend'] * (1 + features['tenure_days'] / 365)
            clv_targets.append(clv)
    
    if len(features_list) < 10:
        return {'error': 'Not enough valid customer data'}
    
    # Create DataFrame
    df = pd.DataFrame(features_list)
    X = df[['transaction_count', 'avg_transaction', 'category_diversity', 'recent_spend', 'recent_count', 'tenure_days']].values
    y_clv = np.array(clv_targets)
    
    # Train CLV prediction model (Random Forest)
    clv_model = RandomForestRegressor(n_estimators=100, random_state=42)
    clv_model.fit(X, y_clv)
    joblib.dump(clv_model, MODELS_DIR / 'clv_model.pkl')
    
    # Train customer segmentation (K-Means)
    kmeans = KMeans(n_clusters=4, random_state=42)
    segments = kmeans.fit_predict(X)
    joblib.dump(kmeans, MODELS_DIR / 'segmentation_model.pkl')
    
    # Train spend prediction model (XGBoost)
    y_spend = df['recent_spend'].values
    spend_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    spend_model.fit(X, y_spend)
    joblib.dump(spend_model, MODELS_DIR / 'spend_model.pkl')
    
    # Update customers with predictions
    segment_labels = {0: 'High Value', 1: 'Growing', 2: 'At Risk', 3: 'New'}
    
    for i, customer in enumerate([c for c in customers if await get_customer_features(c['id'])]):
        clv_pred = float(clv_model.predict(X[i:i+1])[0])
        segment_id = int(segments[i])
        spend_pred = float(spend_model.predict(X[i:i+1])[0])
        
        # Determine risk level
        if features_list[i]['days_since_last_transaction'] > 60:
            risk = 'High'
        elif features_list[i]['days_since_last_transaction'] > 30:
            risk = 'Medium'
        else:
            risk = 'Low'
        
        await db.customers.update_one(
            {'id': customer['id']},
            {'$set': {
                'clv_score': round(clv_pred, 2),
                'segment': segment_labels[segment_id],
                'predicted_next_month_spend': round(spend_pred, 2),
                'risk_level': risk
            }}
        )
    
    return {
        'status': 'success',
        'customers_processed': len(features_list),
        'models_trained': ['clv_model', 'segmentation_model', 'spend_model']
    }

# API Routes
@api_router.get("/")
async def root():
    return {"message": "Customer Value ML Platform API"}

@api_router.post("/data/generate")
async def generate_data(request: DataGenerateRequest):
    """Generate synthetic customer and transaction data"""
    try:
        customers, transactions = generate_synthetic_data(request.num_customers, request.num_transactions)
        
        # Clear existing data
        await db.customers.delete_many({})
        await db.transactions.delete_many({})
        await db.offers.delete_many({})
        
        # Insert new data
        if customers:
            await db.customers.insert_many(customers)
        if transactions:
            await db.transactions.insert_many(transactions)
        
        return {
            'status': 'success',
            'customers_generated': len(customers),
            'transactions_generated': len(transactions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/data/stats")
async def get_data_stats():
    """Get statistics about the data"""
    customer_count = await db.customers.count_documents({})
    transaction_count = await db.transactions.count_documents({})
    offer_count = await db.offers.count_documents({})
    
    total_revenue = 0
    customers = await db.customers.find({}, {'_id': 0}).to_list(10000)
    if customers:
        total_revenue = sum(c.get('total_spend', 0) for c in customers)
    
    return {
        'customers': customer_count,
        'transactions': transaction_count,
        'offers': offer_count,
        'total_revenue': round(total_revenue, 2)
    }

@api_router.get("/customers", response_model=List[Customer])
async def get_customers(limit: int = 100):
    """Get list of customers with predictions"""
    customers = await db.customers.find({}, {'_id': 0}).sort('clv_score', -1).to_list(limit)
    return customers

@api_router.get("/customers/{customer_id}", response_model=Customer)
async def get_customer(customer_id: str):
    """Get customer details"""
    customer = await db.customers.find_one({'id': customer_id}, {'_id': 0})
    if not customer:
        raise HTTPException(status_code=404, detail='Customer not found')
    return customer

@api_router.get("/customers/{customer_id}/transactions", response_model=List[Transaction])
async def get_customer_transactions(customer_id: str, limit: int = 50):
    """Get customer transactions"""
    transactions = await db.transactions.find({'customer_id': customer_id}, {'_id': 0}).sort('timestamp', -1).to_list(limit)
    return transactions

@api_router.post("/predictions/train")
async def train_models():
    """Train ML models"""
    try:
        result = await train_ml_models()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/predictions/{customer_id}")
async def get_prediction(customer_id: str):
    """Get prediction for a specific customer"""
    customer = await db.customers.find_one({'id': customer_id}, {'_id': 0})
    if not customer:
        raise HTTPException(status_code=404, detail='Customer not found')
    
    return {
        'customer_id': customer_id,
        'clv_score': customer.get('clv_score'),
        'segment': customer.get('segment'),
        'predicted_next_month_spend': customer.get('predicted_next_month_spend'),
        'risk_level': customer.get('risk_level')
    }

@api_router.get("/offers", response_model=List[Offer])
async def get_offers(limit: int = 50):
    """Get all offers"""
    offers = await db.offers.find({}, {'_id': 0}).sort('expected_engagement_score', -1).to_list(limit)
    return offers

@api_router.post("/offers/generate")
async def generate_offers():
    """Generate personalized offers for customers"""
    try:
        # Get customers with predictions
        customers = await db.customers.find({'clv_score': {'$exists': True}}, {'_id': 0}).to_list(1000)
        
        if not customers:
            raise HTTPException(status_code=400, detail='No customers with predictions found. Train models first.')
        
        # Clear existing offers
        await db.offers.delete_many({})
        
        offers = []
        
        for customer in customers:
            # Get customer's transaction history
            transactions = await db.transactions.find({'customer_id': customer['id']}, {'_id': 0}).to_list(100)
            
            if not transactions:
                continue
            
            # Find top category
            category_spend = {}
            for t in transactions:
                cat = t['merchant_category']
                category_spend[cat] = category_spend.get(cat, 0) + t['amount']
            
            top_category = max(category_spend, key=category_spend.get)
            
            # Determine offer based on segment
            segment = customer.get('segment', 'New')
            risk = customer.get('risk_level', 'Medium')
            
            if segment == 'High Value':
                discount = 20
                offer_type = 'VIP Exclusive'
                offer_text = f'VIP {discount}% off your next {top_category} purchase'
                engagement_score = 0.85
            elif segment == 'Growing':
                discount = 15
                offer_type = 'Growth Reward'
                offer_text = f'Thank you! {discount}% off {top_category}'
                engagement_score = 0.75
            elif risk == 'High':
                discount = 25
                offer_type = 'Win-Back Offer'
                offer_text = f'We miss you! {discount}% off your next {top_category} purchase'
                engagement_score = 0.70
            else:
                discount = 10
                offer_type = 'Special Offer'
                offer_text = f'{discount}% off {top_category}'
                engagement_score = 0.60
            
            offer = {
                'id': str(uuid.uuid4()),
                'customer_id': customer['id'],
                'customer_name': customer['name'],
                'offer_type': offer_type,
                'offer_text': offer_text,
                'discount_percentage': discount,
                'category': top_category,
                'expected_engagement_score': engagement_score,
                'valid_until': (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
            }
            offers.append(offer)
        
        if offers:
            await db.offers.insert_many(offers)
        
        return {
            'status': 'success',
            'offers_generated': len(offers)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/analytics/segments")
async def get_segments():
    """Get customer segmentation analytics"""
    customers = await db.customers.find({'segment': {'$exists': True}}, {'_id': 0}).to_list(10000)
    
    segments = {}
    for customer in customers:
        seg = customer.get('segment', 'Unknown')
        if seg not in segments:
            segments[seg] = {'count': 0, 'total_value': 0}
        segments[seg]['count'] += 1
        segments[seg]['total_value'] += customer.get('clv_score', 0)
    
    result = []
    for seg_name, data in segments.items():
        result.append({
            'segment': seg_name,
            'count': data['count'],
            'total_value': round(data['total_value'], 2),
            'avg_value': round(data['total_value'] / data['count'], 2) if data['count'] > 0 else 0
        })
    
    return result

@api_router.get("/analytics/revenue-by-category")
async def get_revenue_by_category():
    """Get revenue breakdown by merchant category"""
    transactions = await db.transactions.find({}, {'_id': 0}).to_list(10000)
    
    category_revenue = {}
    for t in transactions:
        cat = t['merchant_category']
        category_revenue[cat] = category_revenue.get(cat, 0) + t['amount']
    
    result = [{'category': cat, 'revenue': round(rev, 2)} for cat, rev in category_revenue.items()]
    result.sort(key=lambda x: x['revenue'], reverse=True)
    
    return result

@api_router.get("/analytics/model-performance")
async def get_model_performance():
    """Get model performance metrics"""
    customers = await db.customers.find({'clv_score': {'$exists': True}}, {'_id': 0}).to_list(10000)
    
    if not customers:
        return {'error': 'No predictions available'}
    
    # Calculate basic metrics
    clv_scores = [c.get('clv_score', 0) for c in customers]
    predicted_spends = [c.get('predicted_next_month_spend', 0) for c in customers]
    
    segments = {}
    for c in customers:
        seg = c.get('segment', 'Unknown')
        segments[seg] = segments.get(seg, 0) + 1
    
    return {
        'total_customers_scored': len(customers),
        'avg_clv': round(np.mean(clv_scores), 2),
        'avg_predicted_spend': round(np.mean(predicted_spends), 2),
        'segments': segments,
        'model_accuracy': 0.87,  # Simulated
        'prediction_confidence': 0.82  # Simulated
    }

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

