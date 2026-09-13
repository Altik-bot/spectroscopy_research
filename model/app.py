import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


from inference_2 import InferenceEngine

engine = InferenceEngine("spectra_model_full.pth")

st.title("Spectra Inference GUI")

spec_file = st.file_uploader("Upload spectrum (.npy)", type=["npy"])
label_file = st.file_uploader("Upload ground truth (.npy)", type=["npy"])

if spec_file is not None:
    spectrum = np.load(spec_file)

    fig, ax = plt.subplots()
    ax.plot(spectrum)
    ax.set_title("Spectrum preview")

    st.pyplot(fig)

    if st.button("Run inference"):
        pred = engine.predict(spectrum)

        st.write("Prediction")
        st.write("log_metallicity:", float(pred[0]))
        st.write("temperature:", float(pred[1]))
        st.write("log_cloud_pressure:", float(pred[2]))

        if label_file is not None:
            gt = np.load(label_file)

            st.write("Ground truth")
            st.write("log_metallicity:", float(gt[0]))
            st.write("temperature:", float(gt[1]))
            st.write("log_cloud_pressure:", float(gt[2]))

            err = np.abs(pred - gt)
            st.write("Absolute error:", err.tolist())