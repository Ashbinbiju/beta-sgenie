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
    
    # Add Order Section
    selected_friend = st.selectbox("Select Friend", [f["name"] for f in data["friends"]])
    selected_items = st.multiselect("Select Items", [i["name"] for i in data["items"]])
    
    if st.button("Add Order", key="add_order"):
        if selected_friend and selected_items:
            for item_name in selected_items:
                item_price = next(item["price"] for item in data["items"] if item["name"] == item_name)
                data["orders"].append({"friend": selected_friend, "item": item_name, "price": item_price})
            save_data(data)
            st.success(f"Added order(s) for {selected_friend}")
            # Clear selected values
            st.session_state.selected_friend = None
            st.session_state.selected_items = []
        else:
            st.error("Please select a friend and at least one item.")

    # Today's Orders Section
    st.subheader("Today's Orders")
    item_quantities = {}
    for order in data["orders"]:
        item_quantities[order["item"]] = item_quantities.get(order["item"], 0) + 1

    for item, quantity in item_quantities.items():
        st.write(f"{item} x{quantity}")

    # User-Specific Orders Section
    st.subheader("Orders by User")
    user_orders = {}
    for order in data["orders"]:
        if order["friend"] not in user_orders:
            user_orders[order["friend"]] = []
        user_orders[order["friend"]].append(order)

    for user, orders in user_orders.items():
        # Display user name with Edit and Delete buttons in a compact row
        col1, col2, col3 = st.columns([4, 1, 1])  # Adjusted column widths for mobile-friendly layout
        with col1:
            st.write(f"**{user}**")
        with col2:
            edit_button = st.button(f"Edit", key=f"edit_user_{user}")
        with col3:
            delete_button = st.button(f"Delete", key=f"delete_user_{user}")

        # Handle button actions
        if edit_button:
            st.session_state.edit_user = user
        if delete_button:
            if st.checkbox(f"Are you sure you want to delete all orders for {user}?", key=f"confirm_delete_{user}"):
                data["orders"] = [o for o in data["orders"] if o["friend"] != user]
                save_data(data)
                st.rerun()

        for order in orders:
            st.write(f"- {order['item']} (₹{order['price']})")

    # Edit User Orders
    if "edit_user" in st.session_state:
        edit_user = st.session_state.edit_user
        st.subheader(f"Edit Orders for {edit_user}")
        edited_items = st.multiselect("Edit Items", [i["name"] for i in data["items"]], 
                                      default=[o["item"] for o in data["orders"] if o["friend"] == edit_user])
        if st.button("Save Changes", key="save_edit_user"):
            data["orders"] = [o for o in data["orders"] if o["friend"] != edit_user]
            for item_name in edited_items:
                item_price = next(item["price"] for item in data["items"] if item["name"] == item_name)
                data["orders"].append({"friend": edit_user, "item": item_name, "price": item_price})
            save_data(data)
            del st.session_state.edit_user
            st.success(f"Orders updated for {edit_user}")
            st.rerun()

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
        else:
            st.error("Please enter a name.")

    st.subheader("Friends List")
    for friend in data["friends"]:
        st.write(friend["name"])
        if friend.get("pic"):
            st.image(friend["pic"], width=100)

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