import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Error Page",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Define the full HTML with inline CSS for the error page
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        .full-screen-error {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: #00AFF9 url('https://cbwconline.com/IMG/Codepen/Unplugged.png') center/cover no-repeat;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            padding-top: 5vh;
            z-index: 1000;
        }
        .error-heading {
            margin: 0.8em 3rem;
            font-family: 'Roboto', sans-serif;
            font-size: 4em;
            color: white;
        }
        .error-message {
            margin: 0.2em 3rem;
            font-family: 'Roboto', sans-serif;
            font-size: 2em;
            color: white;
        }
        /* Hide Streamlit components */
        #MainMenu, footer, header, .block-container {
            visibility: hidden;
        }
    </style>
</head>
<body>
    <div class="full-screen-error">
        <h1 class="error-heading">Whoops!</h1>
        <p class="error-message">Something went wrong</p>
    </div>
</body>
</html>
"""

# Display the custom HTML
st.markdown(html_content, unsafe_allow_html=True)

# Add a hidden button that can be accessed if needed
with st.container():
    if st.button("Refresh"):
        st.experimental_rerun()
