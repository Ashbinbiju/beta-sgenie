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
    selected_items = st.multiselect("Select Items", [i["name"] for i in data["items"]])
    if st.button("Add Order", key="add_order"):
        if selected_friend and selected_items:
            for item_name in selected_items:
                item_price = next(item["price"] for item in data["items"] if item["name"] == item_name)
                data["orders"].append({"friend": selected_friend, "item": item_name, "price": item_price})
            save_data(data)
            st.success(f"Added order(s) for {selected_friend}")
        else:
            st.error("Please select a friend and at least one item.")

    st.subheader("Today's Orders")
    for i, order in enumerate(data["orders"]):
        col1, col2, col3 = st.columns([3, 3, 2])  # Adjust column widths
        with col1:
            st.write(order["item"])
        with col2:
            st.write(f"₹{order['price']}")
        with col3:
            st.write(order["friend"])
            if st.button("Edit", key=f"edit_order_{i}"):
                st.session_state.edit_index = i
            if st.button("Delete", key=f"delete_order_{i}"):
                data["orders"].pop(i)
                save_data(data)
                st.rerun()

    if "edit_index" in st.session_state:
        edit_index = st.session_state.edit_index
        st.subheader("Edit Order")
        edited_friend = st.selectbox("Edit Friend", [f["name"] for f in data["friends"]], index=[f["name"] for f in data["friends"]].index(data["orders"][edit_index]["friend"]))
        edited_item = st.selectbox("Edit Item", [i["name"] for i in data["items"]], index=[i["name"] for i in data["items"]].index(data["orders"][edit_index]["item"]))
        if st.button("Save Changes", key="save_edit"):
            data["orders"][edit_index]["friend"] = edited_friend
            data["orders"][edit_index]["item"] = edited_item
            data["orders"][edit_index]["price"] = next(item["price"] for item in data["items"] if item["name"] == edited_item)
            save_data(data)
            del st.session_state.edit_index
            st.success("Order updated successfully!")
            st.rerun()

    total_cost = sum(order["price"] for order in data["orders"])
    st.subheader(f"Total Cost: ₹{total_cost:.2f}")

    if st.button("Clear All Orders", key="clear_all_orders"):
        data["orders"] = []
        save_data(data)
        st.warning("All orders cleared.")
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
                # Clear input fields after adding
                st.session_state.friend_name = ""
                st.session_state.friend_pic = None
        else:
            st.error("Please enter a name.")

    st.subheader("Friends List")
    cols = st.columns(2)  # Display 2 friends per row
    for i, friend in enumerate(data["friends"]):
        with cols[i % 2]:  # Cycle through columns
            st.write(friend["name"])
            if friend.get("pic"):
                st.image(friend["pic"], width=100, caption="Click to Edit/Delete")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Edit", key=f"edit_friend_{i}"):
                    st.session_state.edit_friend_index = i
            with col2:
                if st.button("Delete", key=f"delete_friend_{i}"):
                    data["friends"].pop(i)
                    save_data(data)
                    st.rerun()

    if "edit_friend_index" in st.session_state:
        edit_index = st.session_state.edit_friend_index
        st.subheader("Edit Friend")
        edited_name = st.text_input("Edit Name", value=data["friends"][edit_index]["name"], key="edit_friend_name")
        edited_pic = st.file_uploader("Edit Picture", type=["jpg", "png"], key="edit_friend_pic")
        if st.button("Save Changes", key="save_friend_edit"):
            data["friends"][edit_index]["name"] = edited_name
            if edited_pic:
                pic_path = f"{PICS_DIR}/{edited_name}.png"
                with open(pic_path, "wb") as f:
                    f.write(edited_pic.getbuffer())
                data["friends"][edit_index]["pic"] = pic_path
            save_data(data)
            del st.session_state.edit_friend_index
            st.success("Friend updated successfully!")
            st.rerun()

    if st.button("Clear All Friends", key="clear_all_friends"):
        data["friends"] = []
        save_data(data)
        st.warning("All friends cleared.")
        st.rerun()

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
        st.rerun()