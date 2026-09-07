import sys
import logging
import os
import io
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import streamlit as st
import plotly.express as px
import torch

from config import settings
from logger.logger_config import create_log
from address_standardization import s2s_address_standardizer as s2s_standardizer
from pipelines.geocodification_pipeline import geocode_pipeline, GEOCODING_FAILED_MESSAGE

## ----------- Logging -----------

create_log()
logger: logging.Logger = logging.getLogger(__name__)


# Configuración de página
st.set_page_config(
    page_title="Geocodificador S2S - ICESI",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# Función para cargar el modelo en caché (evita recargarlo en memoria en cada rerun)
@st.cache_resource(show_spinner=False)
def load_cached_s2s_model(model_path_str: str):
    abs_path = os.path.abspath(model_path_str)
    if not os.path.exists(abs_path):
        logger.warning(f"Ruta de modelo S2S no existe: {abs_path}")
        return None
    try:
        model = s2s_standardizer.load_model(Path(abs_path))
        logger.info(f"Modelo S2S cargado exitosamente desde: {abs_path}")
        return model
    except Exception as e:
        logger.error(f"Error al cargar modelo S2S desde {abs_path}: {e}", exc_info=True)
        st.error(f"Error al cargar modelo S2S desde {abs_path}: {e}")
        return None


# --- SIDEBAR ---
with st.sidebar:
    st.image("https://thumb.wikimedia.org/wikipedia/commons/thumb/6/68/Logo_universidad_icesi.svg/1280px-Logo_universidad_icesi.svg.png?utm_source=es.wikipedia.org&utm_campaign=index&utm_content=thumbnail", width=180)
    st.markdown("### ⚙️ Configuración")

    # Detección de dispositivo
    device = "cuda (GPU)" if torch.cuda.is_available() else "cpu (Procesador)"
    st.caption(f"🖥️ **Dispositivo detectado:** `{device}`")

    # Estado del modelo S2S
    default_model_path = os.path.abspath(settings.S2S_MODEL_PATH or "./s2s_model")
    model_exists = os.path.exists(default_model_path)

    if model_exists:
        st.success("✅ Carpeta de modelo S2S detectada")
    else:
        st.warning(f"⚠️ Carpeta de modelo no encontrada en `{default_model_path}`")

    use_s2s = st.checkbox(
        "Habilitar rescate con modelo S2S",
        value=model_exists,
        help="Si el estandarizador estático falla por sintaxis o ruido, se utiliza la red neuronal S2S para rescatar la dirección.",
        disabled=not model_exists,
    )

    custom_model_path = st.text_input(
        "Ruta del modelo S2S",
        value=default_model_path,
        help="Directorio que contiene model.safetensors, config.json y tokenizer.",
    )

    st.markdown("---")
    st.markdown("### 🗺️ Contexto Geográfico")
    city_param = st.text_input("Ciudad de búsqueda", value="Cali")
    country_param = st.text_input("Departamento / País", value="Valle del Cauca, Colombia")

    batch_size = st.slider("Tamaño de lote (Batch size S2S)", min_value=16, max_value=256, value=64, step=16)

    st.markdown("---")
    st.markdown(
        """
        **Flujo implementado:**
        1. **Estandarización con Reglas Estáticas**
        2. **Estandarización con Modelo S2S**
        3. **Geocodificación con ArcGIS Geocoder**
        """
    )


# --- CONTENIDO PRINCIPAL ---
st.markdown('<div class="main-title">📍 Geocodificador Inteligente de Direcciones</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Estandarización híbrida (Reglas fijas + Rescate neuronal S2S) y georreferenciación con ArcGIS</div>',
    unsafe_allow_html=True,
)

# Cargar modelo si está habilitado
model_instance = None
if use_s2s:
    with st.spinner("Cargando modelo neuronal S2S en memoria... (esto ocurre una sola vez)"):
        model_instance = load_cached_s2s_model(custom_model_path)
    if model_instance is None:
        st.error("No se pudo cargar el modelo S2S. El proceso continuará solo con el estandarizador estático.")

# Subida de archivo (Drag and drop)
uploaded_file = st.file_uploader(
    "Arrastra y suelta aquí tu archivo Excel o CSV con las direcciones:",
    type=["xlsx", "xls", "csv"],
    help="El archivo debe tener al menos una columna con texto de direcciones.",
)

if uploaded_file is not None:
    try:
        # Lectura del archivo
        file_ext = uploaded_file.name.split(".")[-1].lower()

        if file_ext in ["xlsx", "xls"]:
            df_input = pd.read_excel(uploaded_file)

        else:
            try:
                df_input = pd.read_csv(uploaded_file, sep=";")

                if len(df_input.columns) == 1:
                    uploaded_file.seek(0)
                    df_input = pd.read_csv(uploaded_file, sep=",")

            except Exception:
                uploaded_file.seek(0)
                df_input = pd.read_csv(uploaded_file, sep=",")

        logger.info(f"Archivo cargado en GUI: '{uploaded_file.name}' con {len(df_input)} registros.")
        st.success(f"Archivo cargado con éxito: **{uploaded_file.name}** ({len(df_input)} registros encontrados)")

        # Selección de columna de direcciones
        columns = list(df_input.columns)
        candidate_cols = ["dir_orig", "direccion", "dirección", "address", "DIR_ORIG", "DIRECCION", "Direccion"]
        default_index = 0

        for cand in candidate_cols:
            if cand in columns:
                default_index = columns.index(cand)
                break

        col_addr, col_btn = st.columns([3, 2])

        with col_addr:
            selected_col = st.selectbox(
                "Selecciona la columna que contiene las direcciones originales:",
                options=columns,
                index=default_index,
            )

        with col_btn:
            st.write("")
            st.write("")
            start_btn = st.button("Iniciar Estandarización y Geocodificación", type="primary", width='stretch')

        # Vista previa de datos cargados
        with st.expander("👁️ Vista previa de los datos originales (primeras 5 filas)", expanded=False):
            st.dataframe(df_input.head(5), width='stretch')

        # Procesamiento
        if start_btn:
            logger.info(
                f"Iniciando procesamiento en GUI para archivo '{uploaded_file.name}' | "
                f"Columna: '{selected_col}' | Usar S2S: {use_s2s} | Ciudad: '{city_param}' | País: '{country_param}'"
            )

            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(current, total, msg):
                frac = min(1.0, max(0.0, current / total)) if total > 0 else 0.0
                progress_bar.progress(frac)
                status_text.text(f"⏳ {msg} ({int(frac * 100)}%)")

            status_text.text("⚡ Iniciando estandarización y geocodificación...")

            df_final, metrics = geocode_pipeline(
                data=df_input,
                raw_address_column=selected_col,
                standardized_address_column="direccion_norm",
                method_column="metodo_estandarizacion",
                validated_address_column="validated_address",
                latitude_column="latitud",
                longitude_column="longitud",
                status_column="estado_geocodificacion",
                city=city_param,
                country=country_param,
                model=model_instance,
                use_s2s=use_s2s,
                batch_size=batch_size,
                progress_callback=update_progress,
            )

            elapsed = metrics.get("tiempo_segundos", 0.0)
            progress_bar.progress(1.0)
            status_text.success(f"✅ ¡Proceso completado en {elapsed} segundos!")

            # Guardar en session_state
            st.session_state["df_result"] = df_final
            st.session_state["metrics"] = metrics
            st.session_state["raw_col"] = selected_col

            logger.info(
                f"Proceso completado en GUI en {elapsed}s | "
                f"Total: {metrics.get('total_registros')}, Geocodificadas: {metrics.get('geocodificadas_exito')}, "
                f"Rescatadas S2S: {metrics.get('rescatadas_s2s')}"
            )

    except Exception as e:
        logger.error(f"Error procesando el archivo en GUI: {e}", exc_info=True)
        st.error(f"Error procesando el archivo: {e}")


# --- VISUALIZACIÓN DE RESULTADOS Y DASHBOARD ---
if "df_result" in st.session_state:
    df_res = st.session_state["df_result"]
    metrics = st.session_state["metrics"]
    raw_col = st.session_state.get("raw_col", "dir_orig")

    st.markdown("---")
    tab_dash, tab_map, tab_table, tab_download = st.tabs([
        "📊 Dashboard de Métricas",
        "🗺️ Mapa de Puntos",
        "📋 Explorador de Datos",
        "📥 Descargar Archivo",
    ])

    # === TAB 1: DASHBOARD DE MÉTRICAS ===
    with tab_dash:
        st.markdown("### 📈 Indicadores Clave de Rendimiento (KPIs)")

        total = metrics.get("total_records", metrics.get("total_registros", len(df_res)))
        success_geocoded = metrics.get("geocoded_success", metrics.get("geocodificadas_exito", 0))
        generic_geocoded = metrics.get("generic_geocoding", metrics.get("geocodificacion_generica", 0))
        partial_geocoded = metrics.get("partial_geocoding", metrics.get("geocodificacion_parcial", 0))
        s2s_rescued = metrics.get("rescued_s2s", metrics.get("rescatadas_s2s", 0))
        static_standardized = metrics.get("standardized_static", metrics.get("estandarizadas_estatico", 0))
        failed = metrics.get("unlocatable", metrics.get("no_localizables", 0)) + metrics.get("unstandardized", metrics.get("no_estandarizadas", 0))

        pct_geo = metrics.get("geocoding_success_percentage", metrics.get("porcentaje_geocodificacion", 0.0))
        pct_s2s = metrics.get("s2s_contribution_percentage", metrics.get("porcentaje_s2s_aporte", 0.0))
        pct_std = metrics.get("standardization_percentage", metrics.get("porcentaje_estandarizacion", 0.0))

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Direcciones", f"{total:,}")
        with c2:
            st.metric("Geocodificadas con Éxito", f"{success_geocoded:,}", f"{pct_geo}% del total")
        with c3:
            st.metric("Rescatadas por Modelo S2S", f"{s2s_rescued:,}", f"{pct_s2s}% salvadas por IA", delta_color="normal")
        with c4:
            st.metric("No Localizables / Fallidas", f"{failed:,}", f"{round(100 - pct_geo, 1)}%", delta_color="inverse")

        st.markdown("---")
        st.markdown("### 📊 Gráficos de Distribución del Proceso")

        col_g1, col_g2 = st.columns(2)

        # Gráfico 1: Método de Estandarización
        with col_g1:
            std_counts = df_res["metodo_estandarizacion"].value_counts().reset_index()
            std_counts.columns = ["Método", "Cantidad"]
            color_map_std = {
                "ESTATICO": "#10B981",       # Verde esmeralda
                "S2S_RESCATADO": "#6366F1",  # Violeta / Azul IA
                "FALLIDA": "#EF4444",        # Rojo
            }
            labels_std = {
                "ESTATICO": "Reglas Estáticas (Regex)",
                "S2S_RESCATADO": "Rescate Modelo S2S (IA)",
                "FALLIDA": "Estandarización Fallida",
            }
            std_counts["Etiqueta"] = std_counts["Método"].map(labels_std).fillna(std_counts["Método"])

            fig1 = px.pie(
                std_counts,
                values="Cantidad",
                names="Etiqueta",
                hole=0.45,
                color="Método",
                color_discrete_map=color_map_std,
                title="<b>Método de Estandarización de Direcciones</b>",
            )
            fig1.update_traces(textposition="inside", textinfo="percent+value")
            fig1.update_layout(margin=dict(t=50, b=20, l=20, r=20), height=350)
            st.plotly_chart(fig1, width='stretch')

        # Gráfico 2: Estado de Geocodificación
        with col_g2:
            geo_counts = df_res["estado_geocodificacion"].value_counts().reset_index()
            geo_counts.columns = ["Estado", "Cantidad"]
            color_map_geo = {
                "GEOCODIFICADA": "#059669",             # Verde
                "GEOCODIFICACION_GENERICA": "#8B5CF6",   # Violeta
                "GEOCODIFICACION_PARCIAL": "#EC4899",    # Rosa
                "NO_LOCALIZABLE": "#F59E0B",            # Naranja
                "NO_ESTANDARIZADA": "#DC2626",          # Rojo
            }
            labels_geo = {
                "GEOCODIFICADA": "Geocodificada con Éxito",
                "GEOCODIFICACION_GENERICA": "Geocodificación Genérica",
                "GEOCODIFICACION_PARCIAL": "Geocodificación Parcial",
                "NO_LOCALIZABLE": "No Localizable (ArcGIS)",
                "NO_ESTANDARIZADA": "No Estandarizada",
            }
            geo_counts["Etiqueta"] = geo_counts["Estado"].map(labels_geo).fillna(geo_counts["Estado"])

            fig2 = px.pie(
                geo_counts,
                values="Cantidad",
                names="Etiqueta",
                hole=0.45,
                color="Estado",
                color_discrete_map=color_map_geo,
                title="<b>Resultado Final de Geocodificación</b>",
            )
            fig2.update_traces(textposition="inside", textinfo="percent+value")
            fig2.update_layout(margin=dict(t=50, b=20, l=20, r=20), height=350)
            st.plotly_chart(fig2, width='stretch')

        # Tabla resumen detallada
        st.markdown("#### 📋 Detalle de Métricas")
        failed_std_val = metrics.get("failed_standardization", metrics.get("fallidas_estandarizacion", 0))
        unlocatable_val = metrics.get("unlocatable", metrics.get("no_localizables", 0))
        unstd_val = metrics.get("unstandardized", metrics.get("no_estandarizadas", 0))

        summary_data = {
            "Fase": [
                "Estandarización", "Estandarización", "Estandarización",
                "Geocodificación", "Geocodificación", "Geocodificación", "Geocodificación", "Geocodificación",
            ],
            "Concepto": [
                "Estandarizadas por Reglas Estáticas",
                "Rescatadas por Modelo S2S (IA)",
                "Fallo en Estandarización",
                "Geocodificadas con Coordenadas (Éxito)",
                "Geocodificación Genérica (Solo Ciudad/Depto)",
                "Geocodificación Parcial (Solo un número)",
                "No Localizables en ArcGIS",
                "Sin Geocodificar por Fallo de Estandarización",
            ],
            "Registros": [
                static_standardized, s2s_rescued, failed_std_val,
                success_geocoded, generic_geocoded, partial_geocoded,
                unlocatable_val, unstd_val,
            ],
            "Porcentaje": [
                f"{round((static_standardized/total)*100, 2)}%" if total else "0%",
                f"{round((s2s_rescued/total)*100, 2)}%" if total else "0%",
                f"{round((failed_std_val/total)*100, 2)}%" if total else "0%",
                f"{round((success_geocoded/total)*100, 2)}%" if total else "0%",
                f"{round((generic_geocoded/total)*100, 2)}%" if total else "0%",
                f"{round((partial_geocoded/total)*100, 2)}%" if total else "0%",
                f"{round((unlocatable_val/total)*100, 2)}%" if total else "0%",
                f"{round((unstd_val/total)*100, 2)}%" if total else "0%",
            ],
        }
        st.dataframe(pd.DataFrame(summary_data), width='stretch', hide_index=True)


    # === TAB 2: MAPA DE COORDENADAS ===
    with tab_map:
        st.markdown("### 🗺️ Visualización Espacial de Direcciones Geocodificadas")
        valid_map_df = df_res.dropna(subset=["latitud", "longitud"]).copy()
        valid_map_df = valid_map_df[
            (valid_map_df["latitud"] != GEOCODING_FAILED_MESSAGE) &
            (valid_map_df["longitud"] != GEOCODING_FAILED_MESSAGE)
        ]

        if len(valid_map_df) > 0:
            valid_map_df["latitud"] = pd.to_numeric(valid_map_df["latitud"], errors="coerce")
            valid_map_df["longitud"] = pd.to_numeric(valid_map_df["longitud"], errors="coerce")
            valid_map_df = valid_map_df.dropna(subset=["latitud", "longitud"])

            st.caption(f"Mostrando **{len(valid_map_df)}** puntos con coordenadas válidas:")

            # Mapa nativo de Streamlit
            st.map(
                valid_map_df.rename(columns={"latitud": "lat", "longitud": "lon"}),
                latitude="lat",
                longitude="lon",
                zoom=11,
                width='stretch',
            )
        else:
            st.info("No hay coordenadas válidas para mostrar en el mapa.")


    # === TAB 3: TABLA Y FILTROS ===
    with tab_table:
        st.markdown("### 📋 Explorador de Datos Procesados")

        filter_option = st.selectbox(
            "Filtrar registros por:",
            [
                "Todos los registros",
                "Solo Rescatadas por Modelo S2S",
                "Solo Estandarizadas por Reglas Estáticas",
                "Solo Geocodificadas con Éxito",
                "Solo No Localizables",
                "Solo Fallidas en Estandarización",
            ],
        )

        df_view = df_res.copy()
        if filter_option == "Solo Rescatadas por Modelo S2S":
            df_view = df_view[df_view["metodo_estandarizacion"] == "S2S_RESCATADO"]
        elif filter_option == "Solo Estandarizadas por Reglas Estáticas":
            df_view = df_view[df_view["metodo_estandarizacion"] == "ESTATICO"]
        elif filter_option == "Solo Geocodificadas con Éxito":
            df_view = df_view[df_view["estado_geocodificacion"] == "GEOCODIFICADA"]
        elif filter_option == "Solo No Localizables":
            df_view = df_view[df_view["estado_geocodificacion"] == "NO_LOCALIZABLE"]
        elif filter_option == "Solo Fallidas en Estandarización":
            df_view = df_view[df_view["metodo_estandarizacion"] == "FALLIDA"]

        st.caption(f"Mostrando **{len(df_view)}** de {len(df_res)} registros.")

        display_cols = [
            raw_col,
            "direccion_norm",
            "metodo_estandarizacion",
            "validated_address",
            "latitud",
            "longitud",
            "estado_geocodificacion",
        ]
        available_display_cols = [c for c in display_cols if c in df_view.columns]
        st.dataframe(df_view[available_display_cols], width='stretch')


    # === TAB 4: DESCARGA DE ARCHIVO ===
    with tab_download:
        st.markdown("### 📥 Descargar Resultados Geocodificados")
        st.write("Selecciona el formato en el que deseas exportar los datos resultantes:")

        col_d1, col_d2 = st.columns(2)

        # Descarga en Excel
        with col_d1:
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                df_res.to_excel(writer, index=False, sheet_name="Geocodificacion")
            excel_data = excel_buffer.getvalue()

            st.download_button(
                label="📗 Descargar como Excel (.xlsx)",
                data=excel_data,
                file_name="direcciones_geocodificadas.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width='stretch',
            )

        # Descarga en CSV
        with col_d2:
            csv_data = df_res.to_csv(index=False, sep=";", encoding="utf-8-sig")
            st.download_button(
                label="📄 Descargar como CSV (.csv)",
                data=csv_data,
                file_name="direcciones_geocodificadas.csv",
                mime="text/csv",
                width='stretch',
            )
