import streamlit as st
import json
import os
from datetime import datetime

# Constants
DATA_FILE = "data.json"

# Ensure the data file exists
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w") as file:
        json.dump({"friends": [], "orders": []}, file)

# Load existing data
def load_data():
    try:
        with open(DATA_FILE, "r") as file:
            return json.load(file)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return {"friends": [], "orders": []}

# Save data
def save_data(data):
    try:
        with open(DATA_FILE, "w") as file:
            json.dump(data, file, indent=4)
    except Exception as e:
        st.error(f"Error saving data: {e}")

# Load data
data = load_data()

# App title
st.title("Cafeteria Order Tracker ☕")

# Tab 1: Orders (Main Tab)
st.header("Take Orders")

# Add Friend Section
st.subheader("Add Friend")
friend_name = st.text_input("Enter Friend's Name", key="friend_name")
if st.button("Add Friend"):
    if friend_name:
        if friend_name in [f["name"] for f in data["friends"]]:
            st.error("Friend already exists.")
        else:
            data["friends"].append({"name": friend_name})
            save_data(data)
            st.success(f"Added {friend_name}")
            st.session_state.friend_name = ""  # Clear the input field
    else:
        st.error("Please enter a name.")

# Take Order Section
st.subheader("Take Order")
selected_friend = st.selectbox("Select Friend", [f["name"] for f in data["friends"]])
selected_item = st.text_input("Enter Item Name")
selected_price = st.number_input("Enter Item Price (₹)", min_value=0.0, step=0.5)

if st.button("Add Order"):
    if selected_friend and selected_item and selected_price:
        order = {
            "friend": selected_friend,
            "item": selected_item,
            "price": selected_price,
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        data["orders"].append(order)
        save_data(data)
        st.success(f"Added order: {selected_friend} ordered {selected_item} for ₹{selected_price}")
    else:
        st.error("Please fill in all fields.")

# Display Today's Orders
st.subheader("Today's Orders")
if data["orders"]:
    for order in data["orders"]:
        st.write(f"✅ **{order['friend']}** ordered **{order['item']}** for ₹{order['price']} on {order['datetime']}")
else:
    st.write("No orders yet.")

# Total Cost
total_cost = sum(order["price"] for order in data["orders"])
st.subheader(f"Total Cost: ₹{total_cost:.2f}")

# Clear Orders Button
if st.button("Clear All Orders"):
    data["orders"] = []
    save_data(data)
    st.warning("All orders cleared.")

# Tab 2: Past Orders (Optional)
st.sidebar.header("Past Orders")
if st.sidebar.button("View Past Orders"):
    st.subheader("All Past Orders")
    if data["orders"]:
        for order in data["orders"]:
            st.write(f"📅 **{order['datetime']}**: {order['friend']} ordered {order['item']} for ₹{order['price']}")
    else:
        st.write("No past orders found.")