
import streamlit as st
import pandas as pd
import random
from datetime import datetime

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Doctor & Medical Helper", layout="wide")

QUOTES = [
    "Healing takes time, and asking for help is strength.",
    "Every step towards health matters.",
    "Your body can heal if your mind believes.",
    "You are stronger than this illness.",
    "Health is the real wealth."
]

# ---------------- UTILITIES ----------------
def show_motivation():
    st.info(random.choice(QUOTES))

def show_disclaimer():
    st.warning(
        "⚠️ Educational Use Only — This is NOT a real medical system.\n"
        "Always consult a certified doctor before taking any medicine."
    )

# ---------------- DATA LOADERS ----------------
@st.cache_data
def load_doctor_data():
    try:
        return pd.read_csv("doctor.csv")
    except:
        return pd.DataFrame()

@st.cache_data
def load_medicine_data():
    try:
        df = pd.read_csv("medicine.csv")
        if "availability" not in df.columns:
            df["availability"] = "Available"
        return df
    except:
        return pd.DataFrame()

# ---------------- DISEASE RULE ENGINE ----------------
DISEASE_DB = [
    ("fever cold cough sore throat", "Viral Fever", "Rest, warm fluids", "Paracetamol", "After food every 6 hours"),
    ("chest pain sweating left arm", "Heart Problem", "Emergency visit immediately", "Aspirin (doctor only)", "Emergency"),
    ("loose motion diarrhea dehydration", "Food Poisoning", "Drink ORS", "ORS + Zinc", "After every loose motion"),
    ("breathing wheezing", "Asthma", "Avoid dust and smoke", "Salbutamol inhaler", "As needed"),
    ("sugar thirst urination", "Diabetes", "Avoid sugar", "Metformin", "After food"),
    ("acid burning chest", "Acidity", "Avoid spicy food", "Omeprazole", "Before breakfast"),
    ("itch sneezing allergy", "Allergy", "Avoid allergens", "Cetirizine", "At night"),
    ("headache vomiting light pain", "Migraine", "Dark room rest", "Paracetamol", "When pain starts"),
    ("ear pain discharge", "Ear Infection", "Keep ear dry", "Antibiotic drops", "Doctor decides"),
    ("tooth pain swelling", "Dental Infection", "Consult dentist", "Pain killer", "After food"),
    ("weight loss cough night sweat", "Tuberculosis", "Isolation & care", "Anti-TB drugs", "Daily fixed time"),
    ("urine burning pain", "UTI", "Drink water", "Antibiotics", "After food"),
    ("joint pain swelling", "Arthritis", "Warm compress", "Ibuprofen", "After food"),
    ("rash fever travel", "Chikungunya", "Avoid mosquito", "Paracetamol", "After food"),
    ("vision blur headache", "Blood Pressure", "Reduce salt", "Amlodipine", "Once daily"),
    ("vomiting child fever", "Pediatric Infection", "Hydration", "Syrup medicine", "As prescribed"),
    ("leg swelling breathless", "Heart Failure", "Emergency visit", "Diuretics", "Doctor decides"),
    ("skin patch itching", "Fungal Infection", "Keep skin dry", "Antifungal cream", "Twice daily"),
    ("neck stiffness fever", "Meningitis", "Emergency ICU care", "Hospital IV drugs", "Immediate"),
    ("period missed nausea", "Pregnancy", "Gynecologist consultation", "Folic Acid", "Once daily")
]

def predict_disease(symptoms):
    symptoms = symptoms.lower()
    for keys, disease, precaution, medicine, timing in DISEASE_DB:
        for word in keys.split():
            if word in symptoms:
                return disease, precaution, medicine, timing
    return "Unknown Condition", "Consult doctor immediately", "Doctor will decide", "As prescribed"

# ---------------- MODULE 1: SYMPTOM CHECK ----------------
def symptom_checker():
    st.header("🧠 AI Symptom Checker")
    show_motivation()
    show_disclaimer()

    text = st.text_area("Enter Symptoms")

    if st.button("Predict Disease"):
        disease, precaution, medicine, timing = predict_disease(text)
        st.success(f"Predicted Disease: {disease}")
        st.write("✅ Precautions:", precaution)
        st.write("💊 Suggested Medicine:", medicine)
        st.write("⏰ Medicine Timing:", timing)

# ---------------- MODULE 2: HOSPITAL & DOCTOR (DATASET ONLY) ----------------
def hospital_module(df):
    st.header("🏥 Hospital & Doctor Availability")
    show_motivation()

    if df.empty:
        st.error("Doctor dataset not found.")
        return

    locations = df["location"].unique()
    loc = st.selectbox("Select Hospital", locations)

    loc_df = df[df["location"] == loc]
    st.dataframe(loc_df[["name", "specialization", "fee", "is_available_today"]])

    st.success("🚑 Ambulance Available | Call 108")

# ---------------- MODULE 3: MEDICINE (DATASET ONLY) ----------------
def medicine_module(df):
    st.header("💊 Medicine Availability (From Dataset)")
    show_motivation()
    show_disclaimer()

    name = st.text_input("Enter medicine name")

    if st.button("Check Medicine"):
        if df.empty:
            st.error("Medicine dataset not found.")
            return

        res = df[df["name"].str.lower().str.contains(name.lower())]

        if res.empty:
            st.error("Medicine not found in dataset.")
        else:
            for _, r in res.iterrows():
                st.success(f"Name: {r['name']}")
                st.write("Use:", r["use0"])
                st.write("Substitute:", r["substitute0"])
                st.write("Side Effect:", r["sideEffect0"])

                status = str(r["availability"]).lower()
                if status == "available":
                    st.success("🟢 Status: AVAILABLE")
                elif status == "out of stock":
                    st.error("🔴 Status: OUT OF STOCK")
                else:
                    st.info(f"Status: {r['availability']}")

# ---------------- MODULE 4: PAYMENT & APPOINTMENT ----------------
def payment_module(df):
    st.header("💳 Payment & Appointment")
    show_motivation()

    doctor = st.selectbox("Select Doctor", df["name"].unique())
    row = df[df["name"] == doctor].iloc[0]

    fee = row["fee"]
    st.write(f"Doctor Fee: ₹{fee}")

    payment = st.radio("Payment Method", ["UPI", "Card", "NetBanking", "Cash"])
    if st.button("Pay Now"):
        st.success(f"Payment Successful via {payment}")

    date = st.date_input("Appointment Date")
    if st.button("Book Appointment"):
        st.success(f"Appointment booked with {doctor} on {date}")

# ---------------- MAIN APP ----------------
def main():
    st.sidebar.title("AI Medical Assistant")
    page = st.sidebar.radio("Menu", [
        "AI Symptom Checker",
        "Hospital & Doctor",
        "Medicine Availability",
        "Payment & Appointment",
        "About"
    ])

    doctor_df = load_doctor_data()
    medicine_df = load_medicine_data()

    if page == "AI Symptom Checker":
        symptom_checker()
    elif page == "Hospital & Doctor":
        hospital_module(doctor_df)
    elif page == "Medicine Availability":
        medicine_module(medicine_df)
    elif page == "Payment & Appointment":
        payment_module(doctor_df)
    else:
        st.header("About This Project")
        st.write("AI Medical Assistant using REAL doctor & medicine datasets.")

if __name__ == "__main__":
    main()
