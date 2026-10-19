import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union
from sqlalchemy import create_engine, Engine

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

# Helper: builds a reproducible random starting inventory (40% of catalog items) as a DataFrame.
def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

# Helper: (re)creates all SQLite tables (transactions, quotes, quote_requests, inventory) and seeds cash + stock.
def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

# Helper: inserts one 'stock_orders' (purchase) or 'sales' row into the transactions table; returns its id.
def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

# Helper: returns {item_name: units_on_hand} for every item with positive stock as of a date.
def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

# Helper: returns the net units on hand for ONE item as of a date (single-row DataFrame).
def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

# Helper: estimates when the supplier would deliver an order of `quantity` units placed on a date.
def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

# Helper: returns cash = total sales revenue - total stock purchase cost up to a date.
def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


# Helper: returns cash, inventory value, total assets, per-item stock summary and top sellers as of a date.
def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


# Helper: keyword search over historical quotes (joined with their original requests), newest first.
def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################


# Set up and load your env parameters and instantiate your model.

import re
import sys
import difflib
import traceback
from dataclasses import dataclass, field
from typing import Optional

from smolagents import OpenAIServerModel, ToolCallingAgent, tool
from smolagents.monitoring import LogLevel

dotenv.load_dotenv()

OPENAI_API_BASE = "https://openai.vocareum.com/v1"
MODEL_ID = "gpt-4o-mini"  # cheap model, enough for extraction + short replies


def build_model(**extra_settings) -> OpenAIServerModel:
    """Create an LLM client (Vocareum OpenAI proxy). The key is read from .env and never printed."""
    api_key = os.getenv("UDACITY_OPENAI_API_KEY")
    if not api_key or "PASTE_YOUR_KEY" in api_key:
        raise RuntimeError("UDACITY_OPENAI_API_KEY is missing in .env - add your Vocareum key first.")
    return OpenAIServerModel(
        model_id=MODEL_ID,
        api_base=OPENAI_API_BASE,
        api_key=api_key,
        temperature=0.0,
        **extra_settings,
    )


# ----------------------------------------------------------------------
# Business rules (internal - never shown to customers)
# ----------------------------------------------------------------------

CATALOG = {item["item_name"]: item for item in paper_supplies}

# Share of the catalog price that we pay the supplier when restocking (internal margin rule).
SUPPLIER_COST_RATIO = 0.6
# Cash that must remain after any restock purchase.
MIN_CASH_RESERVE = 5000.0
# Items already on our shelves ship the day after the order.
STANDARD_SHIPPING_DAYS = 1
# Safety stock used for catalog items that are not in the inventory reference table.
DEFAULT_MIN_STOCK_LEVEL = 100

# Customer unit -> number of base units (sheets / pieces).
UNIT_MULTIPLIERS = {
    "sheet": 1, "unit": 1, "piece": 1, "roll": 1, "card": 1, "flyer": 1, "poster": 1,
    "board": 1, "plate": 1, "cup": 1, "napkin": 1, "envelope": 1, "bag": 1, "tag": 1,
    "ream": 500,     # 1 ream = 500 sheets
    "pack": 100,     # 1 pack / packet = 100 pieces
    "packet": 100,
    "box": 5000,     # 1 box = 10 reams
}

# (minimum total units in the order, discount rate) - checked from the largest tier down.
BULK_DISCOUNT_TIERS = [(10000, 0.15), (5000, 0.10), (1000, 0.05)]

# Requested formats that we do not sell at all (checked before catalog matching).
UNSUPPORTED_PRODUCTS = [
    (r"\ba3\b", "We do not carry A3-size paper; our sheet formats are A4, letter and legal size."),
    (r"\ba5\b", "We do not carry A5-size paper; our sheet formats are A4, letter and legal size."),
    (r"balloon", "Balloons are not part of our paper-products catalog."),
    (r"ticket", "We do not print or sell event tickets."),
    (r"cardboard", "We do not sell cardboard; our heaviest stock is cardstock / cover stock."),
]

# Ordered keyword rules: the first rule whose keywords are all present wins.
CATALOG_KEYWORD_RULES = [
    (["washi"], "Decorative adhesive tape (washi tape)"),
    (["adhesive tape"], "Decorative adhesive tape (washi tape)"),
    (["streamer"], "Party streamers"),
    (["envelope"], "Envelopes"),
    (["napkin"], "Paper napkins"),
    (["plate"], "Paper plates"),
    (["disposable cup"], "Disposable cups"),
    (["cup"], "Paper cups"),
    (["table cover"], "Table covers"),
    (["tablecloth"], "Table covers"),
    (["sticky"], "Sticky notes"),
    (["notepad"], "Notepads"),
    (["invitation"], "Invitation cards"),
    (["flyer"], "Flyers"),
    (["party bag"], "Paper party bags"),
    (["name tag"], "Name tags with lanyards"),
    (["lanyard"], "Name tags with lanyards"),
    (["folder"], "Presentation folders"),
    (["banner", "roll"], "Rolls of banner paper (36-inch width)"),
    (["banner"], "Banner paper"),
    (["24x36"], "Large poster paper (24x36 inches)"),
    (["poster board"], "Large poster paper (24x36 inches)"),
    (["250 gsm"], "250 gsm cardstock"),
    (["220 gsm"], "220 gsm poster paper"),
    (["100 lb"], "100 lb cover stock"),
    (["80 lb"], "80 lb text paper"),
    (["cardstock"], "Cardstock"),
    (["card stock"], "Cardstock"),
    (["poster"], "Poster paper"),
    (["glossy"], "Glossy paper"),
    (["matte"], "Matte paper"),
    (["photo"], "Photo paper"),
    (["construction"], "Construction paper"),
    (["kraft"], "Kraft paper"),
    (["butcher"], "Butcher paper"),
    (["wrapping"], "Wrapping paper"),
    (["glitter"], "Glitter paper"),
    (["crepe"], "Crepe paper"),
    (["patterned"], "Patterned paper"),
    (["decorative"], "Decorative paper"),
    (["letterhead"], "Letterhead paper"),
    (["legal"], "Legal-size paper"),
    (["uncoated"], "Uncoated paper"),
    (["heavyweight"], "Heavyweight paper"),
    (["eco"], "Eco-friendly paper"),
    (["recycled"], "Recycled paper"),
    (["bright"], "Bright-colored paper"),
    (["color"], "Colored paper"),
    (["colour"], "Colored paper"),
    (["letter"], "Letter-sized paper"),
    (["8.5"], "Letter-sized paper"),
    (["a4"], "A4 paper"),
    (["printer"], "Standard copy paper"),
    (["printing"], "Standard copy paper"),
    (["copy"], "Standard copy paper"),
    (["white paper"], "Standard copy paper"),
    (["standard"], "Standard copy paper"),
]

# Words that must never reach a customer reply (internal data, tooling, errors).
INTERNAL_TERMS_PATTERN = re.compile(
    r"cash balance|cash position|profit|margin|supplier cost|wholesale|cost ratio|inventory value|"
    r"total assets|financial report|traceback|exception|error|_agent|assess_item_request|build_quote|"
    r"finalize_order|get_order_summary|search_past_quotes|tool call|internal",
    re.IGNORECASE,
)


# ----------------------------------------------------------------------
# Shared order state (the "order book"; tools act on the active request)
# ----------------------------------------------------------------------

@dataclass
class OrderLine:
    """One requested item and the decisions taken for it by the agents."""
    customer_description: str
    requested_quantity: float
    requested_unit: str
    units: int
    catalog_item: Optional[str] = None
    status: str = "pending"            # 'available' | 'rejected'
    reason: str = ""                   # customer-safe explanation
    restocked_units: int = 0
    delivery_date: Optional[str] = None
    unit_price: float = 0.0
    line_subtotal: float = 0.0
    line_total: float = 0.0
    sold: bool = False


@dataclass
class OrderRecord:
    """Everything the team knows about one customer request."""
    request_id: str
    request_date: str
    request_text: str
    needed_by_date: Optional[str] = None
    lines: Dict[str, OrderLine] = field(default_factory=dict)
    quote: Optional[Dict] = None
    sale_completed: bool = False
    internal_notes: List[str] = field(default_factory=list)


ORDER_BOOK: Dict[str, OrderRecord] = {}
# The request currently being processed. Tools act on it, so an LLM can never book on the wrong order.
ACTIVE_REQUEST: Dict[str, Optional[str]] = {"request_id": None}


def get_active_order() -> OrderRecord:
    """Return the order record of the request currently being handled."""
    record = ORDER_BOOK.get(ACTIVE_REQUEST["request_id"] or "")
    if record is None:
        raise ValueError("No customer request is being processed right now.")
    return record


def order_status(record: OrderRecord) -> str:
    """Classify an order as 'fulfilled', 'partially fulfilled' or 'not fulfilled'."""
    if not record.lines:
        return "not fulfilled"
    sold = [line for line in record.lines.values() if line.sold]
    if not sold:
        return "not fulfilled"
    return "fulfilled" if len(sold) == len(record.lines) else "partially fulfilled"


def trace(agent_name: str, message: str) -> None:
    """Print a one-line progress trace so the run shows which agent is doing what."""
    print(f"    [{agent_name}] {message}")


# ----------------------------------------------------------------------
# Pure business-logic helpers (no LLM, unit-testable)
# ----------------------------------------------------------------------

def resolve_catalog_item(customer_description: str) -> tuple:
    """Map a free-text item description to (exact catalog name or None, customer-safe reason)."""
    text = customer_description.lower().replace("”", '"').replace("“", '"')
    compact = re.sub(r"[\s\"'″]|inch(es)?", "", text)

    for pattern, reason in UNSUPPORTED_PRODUCTS:
        if re.search(pattern, text):
            return None, reason

    if "24x36" in compact:
        return "Large poster paper (24x36 inches)", ""
    for keywords, catalog_name in CATALOG_KEYWORD_RULES:
        if all(keyword in text for keyword in keywords):
            return catalog_name, ""

    lowered_names = {name.lower(): name for name in CATALOG}
    close = difflib.get_close_matches(text, list(lowered_names), n=1, cutoff=0.75)
    if close:
        return lowered_names[close[0]], ""
    return None, "This item is not part of our current product catalog."


def normalize_unit(unit: str) -> str:
    """Turn 'Reams', 'packets', 'sheets' ... into the singular key used by UNIT_MULTIPLIERS."""
    cleaned = (unit or "unit").strip().lower()
    if cleaned.endswith("es") and cleaned[:-2] in UNIT_MULTIPLIERS:
        return cleaned[:-2]
    if cleaned.endswith("s") and cleaned[:-1] in UNIT_MULTIPLIERS:
        return cleaned[:-1]
    return cleaned if cleaned in UNIT_MULTIPLIERS else "unit"


def convert_to_base_units(quantity: float, unit: str) -> int:
    """Convert a customer quantity (e.g. 5 reams) to base units (2500 sheets)."""
    return int(round(float(quantity) * UNIT_MULTIPLIERS[normalize_unit(unit)]))


def parse_date(value: str) -> Optional[datetime]:
    """Parse ISO or common written date formats; return None if the text is not a date."""
    value = (value or "").strip()
    for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%m/%d/%Y", "%m/%d/%y", "%d %B %Y"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def extract_needed_by_date(request_text: str) -> Optional[str]:
    """Find a 'Month DD, YYYY' deadline in the raw request text (fallback for the LLM value)."""
    match = re.search(r"(January|February|March|April|May|June|July|August|September|October|"
                      r"November|December)\s+\d{1,2},\s*\d{4}", request_text)
    parsed = parse_date(match.group(0)) if match else None
    return parsed.strftime("%Y-%m-%d") if parsed else None


def resolve_needed_by_date(record: OrderRecord, llm_value: str) -> str:
    """Pick the customer deadline: LLM value, else text regex, else request date + 14 days."""
    if record.needed_by_date:
        return record.needed_by_date
    parsed = parse_date(llm_value)
    if parsed:
        record.needed_by_date = parsed.strftime("%Y-%m-%d")
    else:
        record.needed_by_date = extract_needed_by_date(record.request_text) or (
            datetime.fromisoformat(record.request_date) + timedelta(days=14)
        ).strftime("%Y-%m-%d")
    return record.needed_by_date


def min_stock_level_for(item_name: str) -> int:
    """Return the safety-stock level of an item from the inventory table (default if not stocked)."""
    inventory_df = pd.read_sql(
        "SELECT min_stock_level FROM inventory WHERE item_name = :name", db_engine, params={"name": item_name}
    )
    return int(inventory_df.iloc[0]["min_stock_level"]) if not inventory_df.empty else DEFAULT_MIN_STOCK_LEVEL


def bulk_discount_for(total_units: int) -> tuple:
    """Return (discount rate, customer-facing reason) for the total number of units ordered."""
    for threshold, rate in BULK_DISCOUNT_TIERS:
        if total_units >= threshold:
            return rate, (f"{int(rate * 100)}% bulk discount applied because your order totals "
                          f"{total_units:,} units (our {threshold:,}+ unit tier).")
    return 0.0, (f"No bulk discount: your order totals {total_units:,} units; "
                 f"discounts start at 5% for orders of 1,000+ units.")


def unit_label(item_name: str) -> str:
    """Return 'sheets' for paper items and 'units' for products."""
    return "sheets" if CATALOG[item_name]["category"] in {"paper", "specialty"} else "units"


def format_customer_summary(record: OrderRecord) -> str:
    """Customer-safe plain-text summary of an order: items, prices, discount, total, dates, reasons."""
    lines_out = [f"Order reference: {record.request_id}",
                 f"Order date: {record.request_date}",
                 f"Requested delivery by: {record.needed_by_date or 'not specified'}",
                 f"Status: {order_status(record)}", "", "Items we are supplying:"]
    supplied = [line for line in record.lines.values() if line.sold]
    for line in supplied:
        lines_out.append(
            f"- {line.customer_description} -> {line.catalog_item}: {line.units:,} {unit_label(line.catalog_item)} "
            f"x ${line.unit_price:.2f} = ${line.line_subtotal:,.2f}; delivery on {line.delivery_date}"
        )
    if not supplied:
        lines_out.append("- none")
    if record.quote and supplied:
        lines_out += ["", f"Subtotal: ${record.quote['subtotal']:,.2f}",
                      f"Bulk discount: {record.quote['discount_rate'] * 100:.0f}% "
                      f"(-${record.quote['discount_amount']:,.2f})",
                      f"Discount rationale: {record.quote['discount_reason']}",
                      f"Total due: ${record.quote['total']:,.2f}",
                      f"Pricing basis: {record.quote['pricing_note']}",
                      f"Full order delivered by: {record.quote['delivery_date']}"]
    rejected = [line for line in record.lines.values() if not line.sold]
    if rejected:
        lines_out += ["", "Items we cannot supply:"]
        for line in rejected:
            lines_out.append(f"- {line.customer_description} ({line.units:,} requested): "
                             f"{line.reason or 'We could not confirm this item.'}")
    return "\n".join(lines_out)


def compose_fallback_reply(record: OrderRecord) -> str:
    """Deterministic, polite customer reply used if the LLM reply is missing or unsafe."""
    status = order_status(record)
    opening = {
        "fulfilled": "Thank you for your order - everything you asked for is confirmed.",
        "partially fulfilled": "Thank you for your order - we can supply part of it, details below.",
        "not fulfilled": "Thank you for your request. Unfortunately we cannot fulfil it this time.",
    }[status]
    return (f"Dear Customer,\n\n{opening}\n\n{format_customer_summary(record)}\n\n"
            f"Kind regards,\nBeaver's Choice Paper Company - Sales Team")


def sanitize_customer_reply(reply: str, record: OrderRecord) -> str:
    """Remove lines that leak internal data; fall back to the safe template if too much is lost."""
    if not isinstance(reply, str) or len(reply.strip()) < 40:
        return compose_fallback_reply(record)
    kept = [line for line in reply.splitlines() if not INTERNAL_TERMS_PATTERN.search(line)]
    cleaned = "\n".join(kept).strip()
    if len(cleaned) < 0.6 * len(reply.strip()):
        return compose_fallback_reply(record)
    return cleaned


"""Set up tools for your agents to use, these should be methods that combine the database functions above
 and apply criteria to them to ensure that the flow of the system is correct."""


# Tools for inventory agent

@tool
def get_customer_request() -> str:
    """Returns the original customer request being processed: request id, order date, deadline and full text.

    Returns:
        The request header and the customer's full request text.
    """
    record = get_active_order()
    return (f"request_id: {record.request_id}\norder date: {record.request_date}\n"
            f"deadline: {record.needed_by_date or 'not stated'}\n\nCustomer request:\n{record.request_text}")


@tool
def get_inventory_snapshot(as_of_date: str) -> str:
    """Lists every item currently in stock with its units on hand (uses get_all_inventory).

    Args:
        as_of_date: Date of the request in YYYY-MM-DD format.

    Returns:
        A text list of 'item name: units on hand'.
    """
    inventory = get_all_inventory(as_of_date)
    trace("InventoryAgent", f"inventory snapshot as of {as_of_date}: {len(inventory)} items in stock")
    return "\n".join(f"{name}: {int(units)}" for name, units in sorted(inventory.items())) or "No stock."


@tool
def find_catalog_item(customer_description: str) -> str:
    """Maps a customer's wording (e.g. 'A4 glossy paper') to the exact catalog item name and price.

    Args:
        customer_description: The item as the customer wrote it, without the quantity.

    Returns:
        The exact catalog item name and unit price, or why the item cannot be matched.
    """
    catalog_name, reason = resolve_catalog_item(customer_description)
    if catalog_name is None:
        return f"NO MATCH for '{customer_description}': {reason}"
    return f"'{customer_description}' -> '{catalog_name}' (catalog price ${CATALOG[catalog_name]['unit_price']:.2f} each)"


@tool
def check_item_stock(item_name: str, as_of_date: str) -> str:
    """Returns the units on hand for one exact catalog item on a date (uses get_stock_level).

    Args:
        item_name: Exact catalog item name, e.g. 'Glossy paper'.
        as_of_date: Date in YYYY-MM-DD format.

    Returns:
        Units on hand for the item.
    """
    stock = int(get_stock_level(item_name, as_of_date)["current_stock"].iloc[0])
    return f"{item_name}: {stock} units on hand as of {as_of_date}"


@tool
def check_supplier_lead_time(quantity: int, order_date: str) -> str:
    """Estimates when the supplier would deliver a restock order (uses get_supplier_delivery_date).

    Args:
        quantity: Number of units to order from the supplier.
        order_date: Date the restock order would be placed, YYYY-MM-DD.

    Returns:
        The estimated supplier delivery date.
    """
    return f"Supplier delivery for {quantity} units ordered {order_date}: {get_supplier_delivery_date(order_date, quantity)}"


@tool
def assess_item_request(customer_description: str, quantity: float, unit: str, needed_by_date: str) -> str:
    """Checks one requested item end to end and restocks it if needed and possible.

    Steps: maps the description to a catalog item, converts units (ream = 500 sheets, pack = 100),
    checks stock (get_stock_level), and if stock is short places a supplier order
    (create_transaction 'stock_orders') ONLY if the supplier date (get_supplier_delivery_date)
    is on/before the customer's deadline and cash allows (get_cash_balance). Otherwise the item is
    rejected with a reason. The result is saved on the current order so the quoting agent can price it.

    Args:
        customer_description: The item as the customer wrote it, without the quantity (e.g. 'A4 glossy paper').
        quantity: The number the customer asked for (e.g. 500).
        unit: The customer's unit: sheets, reams, packs, rolls, units, boxes ...
        needed_by_date: The customer's delivery deadline in YYYY-MM-DD format.

    Returns:
        A one-line decision: AVAILABLE (with delivery date) or REJECTED (with reason).
    """
    record = get_active_order()
    if record.quote is not None:  # lines are frozen once priced
        return "LOCKED: this order has already been assessed and quoted; no changes made."
    key = customer_description.strip().lower()
    if key in record.lines:  # idempotent: never restock twice for the same line
        line = record.lines[key]
        return f"ALREADY ASSESSED: {line.customer_description} -> {line.status.upper()} {line.reason}".strip()

    needed_by = resolve_needed_by_date(record, needed_by_date)
    units = convert_to_base_units(quantity, unit)
    line = OrderLine(customer_description=customer_description.strip(), requested_quantity=quantity,
                     requested_unit=unit, units=units)
    record.lines[key] = line

    catalog_name, reason = resolve_catalog_item(customer_description)
    if units <= 0:
        line.status, line.reason = "rejected", "The requested quantity was not valid."
    elif catalog_name is None:
        line.status, line.reason = "rejected", reason
    else:
        line.catalog_item = catalog_name
        line.unit_price = CATALOG[catalog_name]["unit_price"]
        _decide_availability(record, line, needed_by)

    trace("InventoryAgent", f"{line.customer_description} ({units} units) -> {line.catalog_item or 'no match'}: "
                            f"{line.status}{' - ' + line.reason if line.reason else ''}")
    if line.status == "available":
        return (f"AVAILABLE: '{line.customer_description}' = {units} x '{catalog_name}', "
                f"ready for delivery on {line.delivery_date} (deadline {needed_by}).")
    return f"REJECTED: '{line.customer_description}' - {line.reason}"


def _decide_availability(record: OrderRecord, line: OrderLine, needed_by: str) -> None:
    """Core inventory rule: serve from stock, or restock only if the supplier is on time and cash allows."""
    request_date = record.request_date
    stock_on_hand = int(get_stock_level(line.catalog_item, request_date)["current_stock"].iloc[0])
    already_reserved = sum(other.units for other in record.lines.values()
                           if other is not line and other.catalog_item == line.catalog_item
                           and other.status == "available")
    free_stock = stock_on_hand - already_reserved

    if free_stock >= line.units:
        ready_date = (datetime.fromisoformat(request_date) + timedelta(days=STANDARD_SHIPPING_DAYS)).strftime("%Y-%m-%d")
    else:
        shortfall = line.units - max(free_stock, 0)
        reorder_units = shortfall + min_stock_level_for(line.catalog_item)  # also restore safety stock
        ready_date = get_supplier_delivery_date(request_date, reorder_units)
        if ready_date > needed_by:  # a smaller order may have a shorter lead time
            reorder_units = shortfall
            ready_date = get_supplier_delivery_date(request_date, reorder_units)
        if ready_date > needed_by:
            line.status = "rejected"
            line.reason = (f"Our current stock does not cover this quantity and the earliest restock would arrive "
                           f"on {ready_date}, after your {needed_by} deadline.")
            return
        restock_cost = round(reorder_units * line.unit_price * SUPPLIER_COST_RATIO, 2)
        if get_cash_balance(request_date) - restock_cost < MIN_CASH_RESERVE:
            line.status = "rejected"
            line.reason = "We cannot source this quantity from our supplier at the moment."
            record.internal_notes.append(f"restock of {line.catalog_item} blocked by cash reserve rule")
            return
        create_transaction(line.catalog_item, "stock_orders", reorder_units, restock_cost, request_date)
        line.restocked_units = reorder_units
        trace("InventoryAgent", f"restock ordered: {reorder_units} x {line.catalog_item}, arrives {ready_date}")

    if ready_date > needed_by:
        line.status = "rejected"
        line.reason = f"The earliest possible delivery is {ready_date}, after your {needed_by} deadline."
        return
    line.status, line.delivery_date = "available", ready_date


# Tools for quoting agent

@tool
def search_past_quotes(search_terms: list[str], limit: int = 3) -> str:
    """Finds similar historical quotes to keep pricing consistent (uses search_quote_history).

    Args:
        search_terms: One or two short keywords, e.g. ['glossy'] or ['cardstock', 'ceremony'].
        limit: Maximum number of past quotes to return.

    Returns:
        Short summaries of matching past quotes (total amount, order size, event type).
    """
    matches = search_quote_history(search_terms, limit=limit)
    trace("QuotingAgent", f"quote history search {search_terms}: {len(matches)} match(es)")
    if not matches:
        return "No similar past quotes found."
    return "\n".join(f"- ${m['total_amount']} | {m['order_size']} {m['event_type']} order | "
                     f"{str(m['quote_explanation'])[:160]}" for m in matches)


@tool
def build_quote() -> str:
    """Prices every AVAILABLE item of the current request at catalog prices and applies the bulk discount tier.

    Discount tiers on total units: 1,000+ = 5%, 5,000+ = 10%, 10,000+ = 15%.
    The quote is saved on the order (quote_id = 'Q-<request_id>') for the sales agent.

    Returns:
        The quote breakdown with the quote_id, or a note that nothing can be quoted.
    """
    record = get_active_order()
    if record.sale_completed:  # never re-price a booked order
        return f"QUOTE {record.quote['quote_id']} already finalized: total ${record.quote['total']:.2f}."
    available = [line for line in record.lines.values() if line.status == "available"]
    if not available:
        if not record.lines:
            return "NOT READY: no items have been assessed yet - the inventory agent must assess them first."
        return "NO QUOTE: none of the requested items can be supplied."

    total_units = sum(line.units for line in available)
    discount_rate, discount_reason = bulk_discount_for(total_units)
    for line in available:
        line.line_subtotal = round(line.units * line.unit_price, 2)
        line.line_total = round(line.line_subtotal * (1 - discount_rate), 2)
    subtotal = round(sum(line.line_subtotal for line in available), 2)
    total = round(sum(line.line_total for line in available), 2)
    history_hits = len(search_quote_history([available[0].catalog_item.split()[0].lower()], limit=5))
    record.quote = {
        "quote_id": f"Q-{record.request_id}",
        "subtotal": subtotal,
        "discount_rate": discount_rate,
        "discount_amount": round(subtotal - total, 2),
        "discount_reason": discount_reason,
        "total": total,
        "delivery_date": max(line.delivery_date for line in available),
        "pricing_note": ("standard catalog unit prices; discount tiers match how we priced "
                         f"{history_hits} comparable past orders" if history_hits else "standard catalog unit prices"),
    }
    trace("QuotingAgent", f"quote {record.quote['quote_id']}: {total_units} units, "
                          f"{discount_rate:.0%} discount, total ${total:,.2f}")
    breakdown = "\n".join(f"- {line.catalog_item}: {line.units} x ${line.unit_price:.2f} = ${line.line_subtotal:.2f}"
                          for line in available)
    return (f"QUOTE {record.quote['quote_id']}\n{breakdown}\nSubtotal ${subtotal:.2f}; {discount_reason} "
            f"Total ${total:.2f}; delivery by {record.quote['delivery_date']}.")


# Tools for ordering agent (SalesAgent)

@tool
def check_cash_position(as_of_date: str) -> str:
    """INTERNAL: returns the company cash balance on a date (uses get_cash_balance). Never share with customers.

    Args:
        as_of_date: Date in YYYY-MM-DD format.

    Returns:
        The cash balance.
    """
    return f"Cash balance on {as_of_date}: ${get_cash_balance(as_of_date):,.2f}"


@tool
def finalize_order(quote_id: str) -> str:
    """Finalizes a quoted order: re-checks stock (get_stock_level) and deadline, then records one
    'sales' transaction per item (create_transaction). Safe to call twice - it never double-books.

    Args:
        quote_id: The quote id produced by the quoting agent, e.g. 'Q-R3'.

    Returns:
        A confirmation listing what was sold, or why the order could not be finalized.
    """
    record = get_active_order()
    if record.quote is None:
        return "CANNOT FINALIZE: the current request has no quote yet - the quoting agent must build it first."
    if quote_id.strip() != record.quote["quote_id"]:  # LLM passed another reference: use the order's own quote
        record.internal_notes.append(f"finalize_order got reference '{quote_id}', used {record.quote['quote_id']}")
    if record.sale_completed:
        return f"ALREADY FINALIZED: order {record.request_id} was recorded earlier."

    sold_lines = []
    for line in record.lines.values():
        if line.status != "available":
            continue
        stock = int(get_stock_level(line.catalog_item, record.request_date)["current_stock"].iloc[0])
        if stock < line.units or line.delivery_date > record.needed_by_date:
            line.status, line.reason = "rejected", "This item could not be confirmed at the final check."
            continue
        create_transaction(line.catalog_item, "sales", line.units, line.line_total, record.request_date)
        line.sold = True
        sold_lines.append(f"{line.units} x {line.catalog_item} (${line.line_total:.2f})")
    record.sale_completed = True
    if sold_lines:  # keep the quote figures equal to what was actually booked
        booked = [booked_line for booked_line in record.lines.values() if booked_line.sold]
        record.quote["subtotal"] = round(sum(booked_line.line_subtotal for booked_line in booked), 2)
        record.quote["total"] = round(sum(booked_line.line_total for booked_line in booked), 2)
        record.quote["discount_amount"] = round(record.quote["subtotal"] - record.quote["total"], 2)
        record.quote["delivery_date"] = max(booked_line.delivery_date for booked_line in booked)
    trace("SalesAgent", f"order {record.request_id} finalized: {len(sold_lines)} line(s) sold")
    return ("SALE RECORDED: " + "; ".join(sold_lines)) if sold_lines else "NOTHING SOLD: no line passed the final check."


@tool
def financial_health_check(as_of_date: str) -> str:
    """INTERNAL: post-sale health check of cash and inventory (uses generate_financial_report).
    Never share the numbers with customers.

    Args:
        as_of_date: Date in YYYY-MM-DD format.

    Returns:
        Cash, inventory value and items at or below zero stock.
    """
    report = generate_financial_report(as_of_date)
    empty_items = [item["item_name"] for item in report["inventory_summary"] if item["stock"] <= 0]
    trace("SalesAgent", f"health check {as_of_date}: assets ${report['total_assets']:,.2f}")
    return (f"cash ${report['cash_balance']:,.2f}; inventory value ${report['inventory_value']:,.2f}; "
            f"out-of-stock items: {', '.join(empty_items) or 'none'}")


# Tools for the orchestrator

@tool
def get_order_summary() -> str:
    """Returns the customer-safe summary of the current order (items, quantities, unit prices, discount,
    total, delivery date, and reasons for any item we cannot supply). Use it to write the reply.

    Returns:
        The customer-safe order summary.
    """
    return format_customer_summary(get_active_order())


# Tools for the business advisor

@tool
def review_business_performance(as_of_date: str) -> str:
    """INTERNAL: summarises all handled orders, rejection reasons, sales and finances
    (uses generate_financial_report and the transactions table) for business recommendations.

    Args:
        as_of_date: Last date of the period in YYYY-MM-DD format.

    Returns:
        A text digest of performance figures.
    """
    report = generate_financial_report(as_of_date)
    statuses = collections_counter(order_status(record) for record in ORDER_BOOK.values())
    rejected = [line for record in ORDER_BOOK.values() for line in record.lines.values() if not line.sold]
    reasons = collections_counter(
        "not in catalog" if line.catalog_item is None else
        "deadline too tight" if "deadline" in line.reason else "other" for line in rejected)
    restocks = pd.read_sql("SELECT COUNT(*) AS n, COALESCE(SUM(price), 0) AS spend FROM transactions "
                           "WHERE transaction_type = 'stock_orders' AND transaction_date > '2025-01-01T00:00:00'",
                           db_engine).iloc[0]
    unmatched = sorted({line.customer_description for line in rejected if line.catalog_item is None})
    top_sellers = [item for item in report["top_selling_products"] if item["item_name"]]  # skip seed cash row
    return (f"orders: {dict(statuses)}\nrejected lines by reason: {dict(reasons)}\n"
            f"unmatched requested products: {unmatched}\n"
            f"restock orders: {int(restocks['n'])} costing ${restocks['spend']:,.2f}\n"
            f"cash ${report['cash_balance']:,.2f}; inventory ${report['inventory_value']:,.2f}\n"
            f"top sellers: {top_sellers}")


def collections_counter(values) -> Dict[str, int]:
    """Count occurrences of each value (small wrapper to keep tool code readable)."""
    counts: Dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


# Set up your agents and create an orchestration agent that will manage them.

INVENTORY_INSTRUCTIONS = """You are the InventoryAgent of Beaver's Choice Paper Company.
Your only job: decide, item by item, whether the current customer request can be supplied on time.
1. Call get_customer_request to read the customer's full request, order date and deadline.
2. Call get_inventory_snapshot once with the order date.
3. For EVERY item listed in the customer's request (count them first), call assess_item_request with the
   item wording exactly as the customer wrote it (without the quantity), the number, the unit
   (sheets, reams, packs, rolls, units...) and the deadline in YYYY-MM-DD.
   Do not skip, merge or rename items, even ones you think we do not sell.
4. Final answer: one line per item copied from the tool results (AVAILABLE / REJECTED + reason).
Never price items and never record sales."""

QUOTING_INSTRUCTIONS = """You are the QuotingAgent of Beaver's Choice Paper Company.
Your only job: price the AVAILABLE items of the current request.
1. Call build_quote once.
2. If a quote was built, call search_past_quotes once with ONE word from the first quoted item name
   (e.g. ['glossy'] for 'Glossy paper', ['cardstock'] for 'Cardstock') to confirm pricing is consistent.
3. Final answer: the quote_id and the breakdown exactly as build_quote returned it.
Never change stock and never record sales."""

SALES_INSTRUCTIONS = """You are the SalesAgent of Beaver's Choice Paper Company.
Your only job: finalize the quoted order in the database.
1. Call check_cash_position with the order date (internal check only).
2. Call finalize_order once with the quote_id.
3. Call financial_health_check with the order date (internal check only).
4. Final answer: the finalize_order result. Do not include cash or financial numbers."""

ORCHESTRATOR_INSTRUCTIONS = """You are the OrchestratorAgent (customer service lead) of Beaver's Choice Paper Company.
You never check stock, price or book sales yourself - you delegate, then write the customer reply.
Call each team member exactly ONCE, in this order:
1. inventory_agent - task: "Assess every item of the current customer request (order date <date>, deadline <deadline>)."
2. quoting_agent - task: "Build the quote for the current request." (skip if every item was REJECTED)
3. sales_agent - task: "Finalize quote <quote_id> (order date <date>)." (only if a quote_id was produced)
4. get_order_summary.
5. final_answer: a short, friendly letter to the customer built ONLY from the order summary:
   - every supplied item with quantity, unit price and line amount, the discount and why, the total, the delivery date;
   - every item we cannot supply with the exact reason from the summary, and a helpful alternative if obvious;
   - copy all numbers exactly; do not invent items, prices or reasons;
   - start with 'Dear Customer,' and sign as 'Beaver's Choice Paper Company - Sales Team'.
   Never mention cash, costs, margins, profit, inventory value, internal tools, agents or errors."""


def create_agents(model: OpenAIServerModel) -> Dict[str, ToolCallingAgent]:
    """Build the 3 worker agents, the orchestrator that manages them, and the business advisor."""
    inventory_agent = ToolCallingAgent(
        tools=[get_customer_request, get_inventory_snapshot, find_catalog_item, check_item_stock,
               check_supplier_lead_time, assess_item_request],
        model=model, name="inventory_agent", instructions=INVENTORY_INSTRUCTIONS, max_steps=12,
        verbosity_level=LogLevel.OFF,
        description="Reads the current customer request, checks stock for every item, maps names to the catalog, "
                    "restocks from the supplier when on time and affordable, and rejects items it cannot supply.",
    )
    quoting_agent = ToolCallingAgent(
        tools=[search_past_quotes, build_quote],
        model=model, name="quoting_agent", instructions=QUOTING_INSTRUCTIONS, max_steps=6,
        verbosity_level=LogLevel.OFF,
        description="Prices the available items of the current request with catalog prices and bulk discounts, "
                    "using quote history for consistency. Returns a quote_id.",
    )
    sales_agent = ToolCallingAgent(
        tools=[check_cash_position, finalize_order, financial_health_check],
        model=model, name="sales_agent", instructions=SALES_INSTRUCTIONS, max_steps=6,
        verbosity_level=LogLevel.OFF,
        description="Finalizes a quoted order: re-checks stock and deadline and records the sales "
                    "transactions. Give it the quote_id and the order date.",
    )
    orchestrator = ToolCallingAgent(
        tools=[get_order_summary],
        managed_agents=[inventory_agent, quoting_agent, sales_agent],
        # one delegation per step, so the stages always run in order
        model=build_model(parallel_tool_calls=False), name="orchestrator_agent",
        instructions=ORCHESTRATOR_INSTRUCTIONS, max_steps=8,
        verbosity_level=LogLevel.OFF,
    )
    business_advisor = ToolCallingAgent(
        tools=[review_business_performance],
        model=model, name="business_advisor_agent", max_steps=3, verbosity_level=LogLevel.OFF,
        instructions="You are the BusinessAdvisorAgent. Call review_business_performance once, then give "
                     "3-5 concrete, numbered recommendations to raise revenue and efficiency (stock, catalog, "
                     "lead times, pricing). Internal audience only.",
    )
    return {"inventory": inventory_agent, "quoting": quoting_agent, "sales": sales_agent,
            "orchestrator": orchestrator, "advisor": business_advisor}


def estimate_item_count(request_text: str) -> int:
    """Rough count of requested items = quantities in the text, ignoring dates, sizes and percentages."""
    text = re.sub(r"\(Date of request:[^)]*\)", " ", request_text)
    text = re.sub(r"(January|February|March|April|May|June|July|August|September|October|November|December)"
                  r"\s+\d{1,2},\s*\d{4}", " ", text)
    text = re.sub(r"\d+(\.\d+)?\s*[\"”']{0,2}\s*x\s*\d+(\.\d+)?", " ", text)  # 24" x 36", 8.5"x11"
    text = re.sub(r"\d+\s*%|\d+\s*(gsm|lb)\b", " ", text)
    return len(re.findall(r"\b\d{1,3}(?:,\d{3})+\b|\b\d+\b", text))


def complete_missing_steps(agents: Dict[str, ToolCallingAgent], record: OrderRecord) -> bool:
    """Safety net: if the orchestrator skipped a stage, run that worker agent directly. Returns True if used."""
    used = False
    expected_items = estimate_item_count(record.request_text)
    if record.quote is None and len(record.lines) < expected_items:
        assessed = ", ".join(line.customer_description for line in record.lines.values()) or "none"
        agents["inventory"].run(f"Assess every item of the current customer request (order date "
                                f"{record.request_date}). Already assessed: {assessed}. The request lists about "
                                f"{expected_items} items - assess the ones still missing.")
        used = True
    if record.quote is None and any(line.status == "available" for line in record.lines.values()):
        agents["quoting"].run("Build the quote for the current request.")
        used = True
    if record.quote is not None and not record.sale_completed:
        agents["sales"].run(f"Finalize quote {record.quote['quote_id']} (order date {record.request_date}).")
        used = True
    if used:
        record.internal_notes.append("safety net completed a skipped stage")
    return used


def reply_matches_order(reply: str, record: OrderRecord) -> bool:
    """Check that the LLM letter quotes the booked total, so customers never get a wrong amount."""
    if record.quote and record.sale_completed and any(line.sold for line in record.lines.values()):
        total = record.quote["total"]
        return f"{total:,.2f}" in reply or f"{total:.2f}" in reply
    return True


def handle_customer_request(agents: Dict[str, ToolCallingAgent], request_id: str, request_text: str,
                            request_date: str, customer_context: str) -> str:
    """Run one customer request through the orchestrator and return a safe customer reply."""
    record = OrderRecord(request_id=request_id, request_date=request_date, request_text=request_text,
                         needed_by_date=extract_needed_by_date(request_text))
    ORDER_BOOK[request_id] = record
    ACTIVE_REQUEST["request_id"] = request_id
    task = (f"request_id: {request_id}\norder date: {request_date}\n"
            f"deadline: {record.needed_by_date or 'not stated (assume 14 days)'}\n"
            f"customer type: {customer_context}\n\nCustomer request:\n{request_text}")
    trace("Orchestrator", f"delegating {request_id}")
    try:
        reply = str(agents["orchestrator"].run(task))
    except Exception:  # never show internal errors to the customer
        record.internal_notes.append(traceback.format_exc(limit=2))
        reply = ""
    try:
        if complete_missing_steps(agents, record):
            reply = ""  # LLM reply was written before the order was complete - rebuild it
    except Exception:
        record.internal_notes.append(traceback.format_exc(limit=2))
    finally:
        ACTIVE_REQUEST["request_id"] = None
    if not reply_matches_order(reply, record):
        record.internal_notes.append("LLM reply total did not match the booked order - template used")
        reply = ""
    return sanitize_customer_reply(reply, record)


# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios():

    print("Initializing Database...")
    init_database(db_engine)
    ORDER_BOOK.clear()
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date", kind="stable")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    ############
    ############
    ############
    # INITIALIZE YOUR MULTI AGENT SYSTEM HERE
    ############
    ############
    ############
    agents = create_agents(build_model())
    starting_cash = current_cash

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        ############
        ############
        ############
        # USE YOUR MULTI AGENT SYSTEM TO HANDLE THE REQUEST
        ############
        ############
        ############

        response = handle_customer_request(
            agents, request_id=f"R{idx+1}", request_text=request_with_date, request_date=request_date,
            customer_context=f"{row['job']} organizing a {row['event']}",
        )

        # Update state
        report = generate_financial_report(request_date)
        previous_cash = current_cash
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Order status: {order_status(ORDER_BOOK[f'R{idx+1}'])}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )
        results[-1]["_cash_changed"] = abs(current_cash - previous_cash) > 0.005

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Evaluation summary
    statuses = [order_status(record) for record in ORDER_BOOK.values()]
    print("\n===== EVALUATION SUMMARY =====")
    print(f"Requests processed: {len(results)}")
    print(f"Requests that changed the cash balance: {sum(r['_cash_changed'] for r in results)}")
    print(f"Fulfilled (all items): {statuses.count('fulfilled')}")
    print(f"Partially fulfilled: {statuses.count('partially fulfilled')}")
    print(f"Not fulfilled: {statuses.count('not fulfilled')}")
    print(f"Net cash change: ${final_report['cash_balance'] - starting_cash:,.2f}")
    print(f"Safety-net interventions: "
          f"{sum('safety net completed a skipped stage' in r.internal_notes for r in ORDER_BOOK.values())}")

    # Internal business review (5th agent)
    try:
        advice = agents["advisor"].run(f"Review the period ending {final_date} and recommend improvements.")
        print("\n===== BUSINESS ADVISOR RECOMMENDATIONS (internal) =====")
        print(advice)
    except Exception as e:
        print(f"Business advisor unavailable: {type(e).__name__}")

    # Save results
    for r in results:
        r.pop("_cash_changed")
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # LLM replies may contain non-ASCII characters
    results = run_test_scenarios()
