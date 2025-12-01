"""
Simple Real Data Scraper - Fetches from Free APIs
"""
import requests
import pymongo
from datetime import datetime, timedelta
import random
import uuid

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["customer_analytics"]

def fetch_crypto_data():
    """Fetch cryptocurrency prices from CoinGecko (FREE)"""
    try:
        print("📡 Fetching crypto prices from CoinGecko...")
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            'vs_currency': 'usd',
            'order': 'market_cap_desc',
            'per_page': 50,
            'page': 1
        }
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        print(f"✅ Fetched {len(data)} crypto prices")
        return data
    except Exception as e:
        print(f"❌ Error fetching crypto: {e}")
        return []

def fetch_customer_data():
    """Fetch customer profiles from RandomUser (FREE)"""
    try:
        print("📡 Fetching customer profiles from RandomUser...")
        url = "https://randomuser.me/api/"
        params = {'results': 100}
        response = requests.get(url, params=params, timeout=10)
        users = response.json()['results']
        print(f"✅ Fetched {len(users)} customer profiles")
        return users
    except Exception as e:
        print(f"❌ Error fetching customers: {e}")
        return []

def transform_to_customers(raw_users):
    """Transform RandomUser data to customer schema"""
    print("🔄 Transforming customer data...")
    customers = []
    
    for user in raw_users:
        customer = {
            'id': user['login']['uuid'],
            'name': f"{user['name']['first']} {user['name']['last']}",
            'email': user['email'],
            'join_date': user['registered']['date'],
            'total_spend': 0,
            'transaction_count': 0,
            'avg_transaction_value': 0,
            'segment': None,
            'clv_score': None,
            'predicted_next_month_spend': None,
            'risk_level': None
        }
        customers.append(customer)
    
    print(f"✅ Transformed {len(customers)} customers")
    return customers

def create_transactions(crypto_data, customers):
    """Create realistic transactions using crypto prices"""
    print("🔄 Creating transactions from crypto prices...")
    transactions = []
    merchant_categories = [
        'Restaurants', 'Groceries', 'Gas Stations', 'Online Shopping',
        'Entertainment', 'Travel', 'Healthcare', 'Utilities',
        'Electronics', 'Fashion', 'Home Improvement', 'Fitness'
    ]
    
    for customer in customers:
        num_transactions = random.randint(5, 25)
        
        for _ in range(num_transactions):
            crypto = random.choice(crypto_data)
            base_amount = crypto.get('current_price', 100)
            amount = round(abs(base_amount * random.uniform(0.1, 2.0)), 2)
            
            transaction = {
                'id': str(uuid.uuid4()),
                'customer_id': customer['id'],
                'merchant_category': random.choice(merchant_categories),
                'amount': amount,
                'timestamp': (datetime.now() - timedelta(days=random.randint(0, 180))).isoformat(),
                'merchant_name': f"{random.choice(merchant_categories)} Store {random.randint(1, 20)}"
            }
            transactions.append(transaction)
            
            customer['total_spend'] += amount
            customer['transaction_count'] += 1
    
    # Calculate averages
    for customer in customers:
        if customer['transaction_count'] > 0:
            customer['avg_transaction_value'] = round(
                customer['total_spend'] / customer['transaction_count'], 2
            )
    
    print(f"✅ Created {len(transactions)} transactions")
    return transactions, customers

def load_to_mongodb(customers, transactions):
    """Load data into MongoDB"""
    print("💾 Loading data into MongoDB...")
    
    # Clear existing data
    db.customers.delete_many({})
    db.transactions.delete_many({})
    db.offers.delete_many({})
    print("   🗑️  Cleared old data")
    
    # Insert new data
    if customers:
        db.customers.insert_many(customers)
        print(f"   ✅ Inserted {len(customers)} customers")
    
    if transactions:
        db.transactions.insert_many(transactions)
        print(f"   ✅ Inserted {len(transactions)} transactions")
    
    total_revenue = sum(c['total_spend'] for c in customers)
    return len(customers), len(transactions), total_revenue

def main():
    print("\n" + "="*60)
    print("🚀 REAL DATA PIPELINE - STARTING")
    print("="*60 + "\n")
    
    # Step 1: Scrape from APIs
    print("STEP 1: Scraping from Free APIs")
    print("-" * 40)
    crypto_data = fetch_crypto_data()
    raw_users = fetch_customer_data()
    
    if not crypto_data or not raw_users:
        print("\n❌ FAILED: Could not fetch data from APIs")
        print("   Check your internet connection and try again")
        return
    
    # Step 2: Transform data
    print("\nSTEP 2: Transforming Data")
    print("-" * 40)
    customers = transform_to_customers(raw_users)
    transactions, customers = create_transactions(crypto_data, customers)
    
    # Step 3: Load to database
    print("\nSTEP 3: Loading to Database")
    print("-" * 40)
    customer_count, transaction_count, total_revenue = load_to_mongodb(customers, transactions)
    
    # Summary
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE!")
    print("="*60)
    print(f"\n📊 Summary:")
    print(f"   • Customers loaded: {customer_count}")
    print(f"   • Transactions created: {transaction_count}")
    print(f"   • Total revenue: ${total_revenue:,.2f}")
    print(f"   • Data source: Real APIs (CoinGecko + RandomUser)")
    print(f"\n🎯 Next Steps:")
    print(f"   1. Start backend: python main.py")
    print(f"   2. Start frontend: npm start")
    print(f"   3. Train ML models in the Data page")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
