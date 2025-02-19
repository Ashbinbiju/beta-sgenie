import streamlit as st
import json
import os

# Constants
DATA_FILE = "data.json"
PICS_DIR = "pics"

# Ensure the pics directory exists
os.makedirs(PICS_DIR, exist_ok=True)

# Load existing data
def load_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as file:
                data = json.load(file)
                # Ensure all friends have a 'pic' key
                for friend in data.get("friends", []):
                    friend.setdefault("pic", None)
                return data
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return {"friends": [], "items": [], "orders": []}
    return {"friends": [], "items": [], "orders": []}

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
st.title("Cafeteria Order Tracker")

# Tabs for navigation
tab1, tab2, tab3 = st.tabs(["📝 Orders", "📌 Friends", "🍔 Items"])

# Tab 1: Orders
with tab1:
    st.header("Take Orders")
    selected_friend = st.selectbox("Select Friend", [f["name"] for f in data["friends"]])
    selected_item = st.selectbox("Select Item", [i["name"] for i in data["items"]])
    if st.button("Add Order", key="add_order"):
        selected_item_price = next(item["price"] for item in data["items"] if item["name"] == selected_item)
        data["orders"].append({"friend": selected_friend, "item": selected_item, "price": selected_item_price})
        save_data(data)
        st.success(f"Added order: {selected_friend} ordered {selected_item}")

    st.subheader("Today's Orders")
    for i, order in enumerate(data["orders"]):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(f"{order['friend']} ordered {order['item']} - ₹{order['price']}")
        with col2:
            if st.button("Delete", key=f"delete_order_{i}"):
                data["orders"].pop(i)
                save_data(data)
                st.rerun()

    total_cost = sum(order["price"] for order in data["orders"])
    st.subheader(f"Total Cost: ₹{total_cost:.2f}")

    if st.button("Clear All Orders", key="clear_all_orders"):
        data["orders"] = []
        save_data(data)
        st.warning("All orders cleared.")

# Tab 2: Friends
with tab2:
    st.header("Manage Friends")
    name = st.text_input("Enter Friend's Name", key="friend_name")
    pic = st.file_uploader("Upload a Picture", type=["jpg", "png"], key="friend_pic")
    if st.button("Add Friend", key="add_friend"):
        if name:
            if name in [f["name"] for f in data["friends"]]:
                st.error("Friend already exists.")
            else:
                friend = {"name": name, "pic": None}
                if pic:
                    pic_path = f"{PICS_DIR}/{name}.png"
                    with open(pic_path, "wb") as f:
                        f.write(pic.getbuffer())
                    friend["pic"] = pic_path
                data["friends"].append(friend)
                save_data(data)
                st.success(f"Added {name}")
                # Clear input fields after adding
                st.session_state.friend_name = ""
                st.session_state.friend_pic = None
        else:
            st.error("Please enter a name.")

    st.subheader("Friends List")
    cols = st.columns(3)  # Display 3 friends per row
    for i, friend in enumerate(data["friends"]):
        with cols[i % 3]:  # Cycle through columns
            st.write(friend["name"])
            if friend.get("pic"):
                st.image(friend["pic"], width=100)
            if st.button("Delete", key=f"delete_friend_{i}"):
                data["friends"].pop(i)
                save_data(data)
                st.rerun()

    if st.button("Clear All Friends", key="clear_all_friends"):
        data["friends"] = []
        save_data(data)
        st.warning("All friends cleared.")

# Tab 3: Items
with tab3:
    st.header("Cafeteria Items")
    item_name = st.text_input("Item Name", key="item_name")
    item_price = st.number_input("Price", min_value=0.0, step=0.5, key="item_price")
    if st.button("Add Item", key="add_item"):
        if item_name and item_price:
            if item_name in [i["name"] for i in data["items"]]:
                st.error("Item already exists.")
            else:
                data["items"].append({"name": item_name, "price": item_price})
                save_data(data)
                st.success(f"Added {item_name}")
        else:
            st.error("Please fill in all fields.")

    st.subheader("Items List")
    for i, item in enumerate(data["items"]):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(f"{item['name']} - ₹{item['price']}")
        with col2:
            if st.button("Delete", key=f"delete_item_{i}"):
                data["items"].pop(i)
                save_data(data)
                st.rerun()

    if st.button("Clear All Items", key="clear_all_items"):
        data["items"] = []
        save_data(data)
        st.warning("All items cleared.")