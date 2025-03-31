import requests
import streamlit as st

# API URLs for login, signup, and image upload
BASE_URL = "http://localhost:8000"


# Function to handle signup
def signup(email, password):
    payload = {"user_email": email, "user_password": password}
    response = requests.post(f"{BASE_URL}/signup", json=payload)

    if response.status_code == 200:
        st.success("Account created successfully!")
    else:
        try:
            error_detail = response.json().get("detail", "Unknown error")
        except requests.exceptions.JSONDecodeError:
            error_detail = f"Error: {response.text}"

        st.error(f"Error: {error_detail}")


# Function to handle login
def login(email, password):
    payload = {"user_email": email, "user_password": password}
    response = requests.post(f"{BASE_URL}/login", json=payload)
    if response.status_code == 200:
        st.session_state.logged_in = True
        st.session_state.user_email = email
        st.session_state.user_id = response.json().get("user_id")
        st.session_state.page = "upload"

        st.session_state.page_refreshed = True
        raise st.rerun()

    else:
        st.error("Error: " + response.json().get("detail", "Invalid credentials"))


# Streamlit interface for login and signup
def login_signup_page():
    st.title("Login or Signup")

    option = st.radio("Choose an option", ["Login", "Signup"])

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if option == "Signup":
        if st.button("Create Account"):
            if email and password:
                signup(email, password)
            else:
                st.warning("Please fill out all fields.")
    elif option == "Login":
        if st.button("Login"):
            if email and password:
                login(email, password)
            else:
                st.warning("Please fill out all fields.")


def image_upload_page():
    st.title(f"Welcome, {st.session_state.user_email}")
    st.subheader("Upload an Image for Prediction")

    image = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

    if image:
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Auto-upload image
        files = {"image": ("image.jpg", image.getvalue(), image.type)}
        response = requests.post(
            f"{BASE_URL}/upload_image",
            data={"user_id": str(st.session_state.user_id)},
            files=files
        )

        if response.status_code == 200:
            st.success("Image uploaded successfully!")
            st.session_state.image_id = response.json().get("image_id")
            st.session_state.predicted_digit = None

            predict_result()

    if "predicted_digit" in st.session_state and st.session_state.predicted_digit is not None:
        st.subheader("Prediction Result")
        st.write(f"Predicted Digit: {st.session_state.predicted_digit}")


def predict_result():
    response = requests.post(
        f"{BASE_URL}/predict",
        json={"image_id": st.session_state.image_id}
    )

    if response.status_code == 200:
        result = response.json()
        st.session_state.predicted_digit = result["predicted_digit"]
    else:
        st.error("Error predicting the result.")


def main():
    if 'logged_in' in st.session_state and st.session_state.logged_in:
        image_upload_page()
    else:
        login_signup_page()


if __name__ == "__main__":
    main()
