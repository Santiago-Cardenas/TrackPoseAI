# Entrega3/src/front/app.py

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from model.activity_predictor import ActivityPredictor


st.set_page_config(
    page_title="TrackPoseAI - Demo",
    page_icon="🏃",
    layout="centered",
)

st.title("🏃 TrackPoseAI - Clasificador de Actividades Humanas")
st.write(
    "Sube un video (.mp4) de la persona realizando las actividades "
    "(caminar, girar, sentarse, levantarse) y el sistema te dirá "
    "la actividad dominante."
)
st.write(
    "Consentimiento: Al subir un video, aceptas que este será procesado "
    "para la detección de actividades. "
    "Sus videos no serán almacenados ni compartidos y tampoco se utilizarán para el entrenamiento del modelo."
)

uploaded_file = st.file_uploader("📹 Sube un video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    st.video(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        temp_video_path = tmp.name

    st.info("Procesando el video, esto puede tardar unos segundos...")

    try:
        predictor = ActivityPredictor()
        preds, probs = predictor.predict(temp_video_path)
        summary = predictor.get_summary(preds)

        st.subheader("✅ Actividad detectada")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Actividad dominante", summary["dominant_activity"])
        with col2:
            st.metric(
                "Confianza",
                f"{summary['dominant_percentage']:.1f} %",
            )

        st.subheader("📊 Distribución de actividades (porcentaje de frames)")

        dist_df = (
            pd.DataFrame(
                {
                    "Actividad": list(summary["summary"].keys()),
                    "Porcentaje": list(summary["summary"].values()),
                }
            )
            .sort_values("Porcentaje", ascending=False)
            .set_index("Actividad")
        )

        st.bar_chart(dist_df)

        st.subheader("⏱️ Segmentos de actividad en el tiempo")

        window_size = 30

        segmentos = []
        import math
        from collections import Counter

        n_frames = len(preds)
        for start in range(0, n_frames, window_size):
            end = min(start + window_size, n_frames)
            chunk = preds[start:end]

            counts = Counter(chunk)
            dom_label, dom_count = counts.most_common(1)[0]
            porcentaje = 100.0 * dom_count / len(chunk)

            segmentos.append({
                "Segmento": len(segmentos) + 1,
                "Frame inicio": start,
                "Frame fin": end - 1,
                "Actividad dominante": dom_label,
            })

        segmentos_df = pd.DataFrame(segmentos)
        st.dataframe(segmentos_df)

        st.caption(
            "Cada fila representa un tramo del video. "
            "La columna 'Actividad dominante' muestra la clase principal en ese tramo, "
            "y 'Porcentaje en el segmento' indica qué tan consistente fue."
        )



    except Exception as e:
        st.error(f"Ocurrió un error al procesar el video: {e}")

    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
