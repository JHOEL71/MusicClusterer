""" 
Aplicación de despliegue — Segmentación de canciones por perfil de audio (K-Means) 
Proyecto Final — Sistemas Inteligentes (Machine Learning) — UNC 
""" 

import joblib 
import numpy as np 
import pandas as pd 
import streamlit as st 
import matplotlib.pyplot as plt 
import yt_dlp  # Asegúrate de instalar con: pip install yt-dlp

# --------------------------------------------------------------------------- 
# Configuración de página 
# --------------------------------------------------------------------------- 
st.set_page_config( 
    page_title="Music Clusterer | UNC & Spotify", 
    page_icon="🎵", 
    layout="wide", 
    initial_sidebar_state="expanded", 
) 

# --------------------------------------------------------------------------- 
# Paleta de colores por cluster
# --------------------------------------------------------------------------- 
CLUSTER_COLORS = { 
    0: "#FF0055",  # Neon Pink / Red
    1: "#00F5D4",  # Electric Turquoise
    2: "#FFB703",  # Amber Gold
    3: "#B5179E",  # Deep Purple / Magenta
    4: "#1DB954",  # Spotify Green
} 

PERFILES = { 
    0: ("Intenso / Alta Energía", "Distorsión, ritmo rápido y poca acústica.", "Rock · Metal · Hardcore"), 
    1: ("Acústico & Instrumental", "Perfil suave, melódico e instrumental relajante.", "Ambient · Clásica · Lo-Fi"), 
    2: ("Pop / Dance Positivo", "Música pegajosa, alta valencia y brillantez sonora.", "Pop · Electropop · Top 40"), 
    3: ("Acústico Melódico", "Enfoque en voz y guitarra/piano, orgánico y expresivo.", "Baladas · Folk · Indie"), 
    4: ("Bailable & Festivo", "Ritmos marcados, ideales para pista de baile.", "Reggaetón · Salsa · EDM"), 
} 

NOMBRES_CORTOS_CLUSTERS = {
    0: "Intenso / Energía",
    1: "Acústico / Instrumental",
    2: "Pop / Positivo",
    3: "Acústico Melódico",
    4: "Bailable & Festivo",
}

# --------------------------------------------------------------------------- 
# Estilos UI Spotify
# --------------------------------------------------------------------------- 
st.markdown( 
    """ 
    <style> 
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { 
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background: #080B10;
    }

    footer, header { visibility: hidden !important; }
    
    /* ---- Hero Header ---- */
    .hero {
        position: relative;
        padding: 30px 36px;
        border-radius: 20px;
        margin-bottom: 24px;
        background: linear-gradient(135deg, rgba(29, 185, 84, 0.25) 0%, rgba(18, 18, 18, 0.8) 60%, rgba(8, 11, 16, 0.95) 100%);
        border: 1px solid rgba(29, 185, 84, 0.35);
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.6);
    }
    
    .hero-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 30px;
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 1px;
        text-transform: uppercase;
        background: rgba(29, 185, 84, 0.15);
        color: #1DB954;
        border: 1px solid rgba(29, 185, 84, 0.4);
        margin-bottom: 12px;
    }

    .hero h1 {
        font-size: 2.2rem;
        font-weight: 800;
        color: #FFFFFF;
        margin: 0;
        letter-spacing: -1px;
    }

    .hero p {
        color: #A7A7A7;
        margin: 8px 0 0 0;
        font-size: 0.95rem;
        font-weight: 500;
    }

    /* ---- Tarjetas Neumórficas ---- */
    .cluster-card {
        border-radius: 20px;
        padding: 24px;
        margin: 20px 0;
        background: linear-gradient(145deg, #121824 0%, #0D121D 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 6px solid var(--accent, #1DB954);
        box-shadow: 0 12px 30px rgba(0,0,0,0.5);
    }

    .cluster-card h3 {
        font-size: 1.35rem;
        font-weight: 800;
        margin: 10px 0;
        color: #FFFFFF;
    }

    .cluster-card .badge {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 0.8px;
        color: #000000;
        background: var(--accent, #1DB954);
    }

    .cluster-card p { color: #CBD5E1; margin: 6px 0; font-size: 0.95rem; }
    
    .cluster-card .genres { 
        color: #94A3B8; 
        font-size: 0.88rem; 
        font-weight: 600;
        margin-top: 12px;
        padding-top: 12px;
        border-top: 1px solid rgba(255,255,255,0.08);
    }

    /* ---- Métricas ---- */
    .metric-box {
        background: #121824;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 16px 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }

    .metric-box .label { 
        color: #64748B; 
        font-size: 0.72rem; 
        font-weight: 800;
        text-transform: uppercase; 
        letter-spacing: 1px; 
        margin-top: 4px;
    }

    .metric-box .value { 
        color: #FFFFFF; 
        font-size: 1.5rem; 
        font-weight: 800; 
        line-height: 1.2;
    }

    /* ---- Player Card ---- */
    .player-card {
        background: linear-gradient(135deg, #121824 0%, #0A0E17 100%);
        border: 1px solid rgba(29, 185, 84, 0.4);
        border-radius: 20px;
        padding: 22px;
        margin-bottom: 20px;
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background-color: #05070A !important;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    .section-title {
        font-size: 1.1rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-bottom: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Carga de artefactos y utilidades
# ---------------------------------------------------------------------------
def construir_vector_aux(row_dict, scaler, config):
    num_cols = config["num_cols"]
    bin_cols = config["bin_cols"]
    cat_cols = config["cat_cols"]
    feature_names = config["feature_names"]

    num_df = pd.DataFrame([{c: row_dict.get(c, 0) for c in num_cols}])
    num_scaled = pd.DataFrame(scaler.transform(num_df), columns=num_cols)
    bin_df = pd.DataFrame([{c: int(row_dict.get(c, 0)) for c in bin_cols}])

    cat_presentes = [c for c in cat_cols if c in row_dict]
    if cat_presentes:
        cat_row = {c: str(row_dict[c]) for c in cat_presentes}
        cat_df = pd.DataFrame([cat_row])
        cat_dummies = pd.get_dummies(cat_df, prefix=cat_presentes)
    else:
        cat_dummies = pd.DataFrame()

    X = pd.concat([num_scaled, bin_df, cat_dummies], axis=1)
    return X.reindex(columns=feature_names, fill_value=0)


@st.cache_resource
def cargar_artefactos():
    scaler = joblib.load("artifacts/scaler.pkl")
    config = joblib.load("artifacts/feature_config.pkl")
    kmeans = joblib.load("artifacts/kmeans_model.pkl")
    pca = joblib.load("artifacts/pca_visualizacion.pkl")
    df = pd.read_csv("artifacts/dataset_clusterizado.csv")

    if "pca_1" not in df.columns or "pca_2" not in df.columns:
        vectores = [construir_vector_aux(fila.to_dict(), scaler, config) for _, fila in df.iterrows()]
        X_full = pd.concat(vectores, ignore_index=True)
        pca_coords = pca.transform(X_full)
        df["pca_1"] = pca_coords[:, 0]
        df["pca_2"] = pca_coords[:, 1]

    return scaler, config, kmeans, pca, df


scaler, config, kmeans, pca, df = cargar_artefactos()

def construir_vector(row_dict):
    return construir_vector_aux(row_dict, scaler, config)

@st.cache_data(ttl=3600)
def obtener_metadatos_youtube(url):
    """Extrae metadatos del video de Youtube y busca la mejor coincidencia en la BD."""
    try:
        ydl_opts = {'quiet': True, 'skip_download': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get('title', '')
            artist = info.get('artist', '') or info.get('uploader', '')
            
            # Buscar en el dataset si el título o artista coincide
            coincidencias = df[
                df['track_name'].str.contains(title[:15], case=False, na=False) |
                df['artists'].str.contains(artist[:15], case=False, na=False)
            ]
            
            if not coincidencias.empty:
                return title, artist, coincidencias.iloc[0].to_dict()
            return title, artist, None
    except Exception:
        return "Video no identificado", "Desconocido", None


# ---------------------------------------------------------------------------
# Gráfico PCA
# ---------------------------------------------------------------------------
def graficar_pca(punto_pca=None, cluster_pred=None):
    fig, ax = plt.subplots(figsize=(6.5, 5.8), facecolor="#121824")
    ax.set_facecolor("#080B10")

    clusters_existentes = sorted(df["cluster_kmeans"].unique())
    for c_id in clusters_existentes:
        sub_df = df[df["cluster_kmeans"] == c_id]
        ax.scatter(
            sub_df["pca_1"],
            sub_df["pca_2"],
            c=CLUSTER_COLORS[c_id],
            s=22,
            alpha=0.4,
            linewidths=0,
            label=f"C{c_id}: {NOMBRES_CORTOS_CLUSTERS.get(c_id, f'Cluster {c_id}')}",
        )

    if punto_pca is not None:
        accent = CLUSTER_COLORS.get(cluster_pred, "#1DB954")
        
        ax.scatter(
            punto_pca[0],
            punto_pca[1],
            c=accent,
            s=180,
            alpha=0.3,
            linewidths=0,
            zorder=4,
        )
        ax.scatter(
            punto_pca[0],
            punto_pca[1],
            c="#FFFFFF",
            marker="*",
            s=80,
            edgecolor=accent,
            linewidth=1.2,
            label="★ Tu Canción",
            zorder=5,
        )

    ax.set_xlabel("Componente Principal 1", color="#94A3B8", fontsize=9, fontweight="700", labelpad=8)
    ax.set_ylabel("Componente Principal 2", color="#94A3B8", fontsize=9, fontweight="700", labelpad=8)
    ax.tick_params(colors="#64748B", labelsize=8)

    for spine in ax.spines.values():
        spine.set_color("#1E293B")
        spine.set_linewidth(1.0)

    ax.grid(True, color="#1E293B", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_title("Espacio Latente PCA — Segmentación de Clusters", color="#F8FAFC", fontsize=11, pad=14, fontweight="bold")

    ax.legend(
        loc="lower left",
        ncol=2,
        facecolor="#0D121D",
        edgecolor="#2A3140",
        labelcolor="#F8FAFC",
        fontsize=8,
        markerscale=1.0,
        framealpha=0.9,
        borderpad=0.6,
        handletextpad=0.5,
        columnspacing=0.8,
    )

    plt.tight_layout()
    return fig


def metric_box(label, value):
    st.markdown(
        f"""<div class="metric-box">
            <div class="value">{value}</div>
            <div class="label">{label}</div>
        </div>""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Encabezado Principal
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <span class="hero-badge">Spotify Audio Intelligence</span>
        <h1>Music Clusterer Dashboard</h1>
        <p>Segmentación y perfilado de canciones por machine learning</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Panel Lateral (Sidebar)
# ---------------------------------------------------------------------------
st.sidebar.title("🎛️ Control Panel")

modo = st.sidebar.radio("Origen de datos:", ["Catálogo preexistente", "Ingreso por URL / Atributos Manuales"])
valores = {}

if modo == "Catálogo preexistente":
    st.sidebar.markdown("### 🔍 Selección de track")
    if "label_busqueda" not in df.columns:
        df["label_busqueda"] = df["track_name"].astype(str) + " — " + df["artists"].astype(str)
        
    opciones = df["label_busqueda"].dropna().unique().tolist()
    nombre = st.sidebar.selectbox("Buscar canción:", opciones)
    
    fila = df[df["label_busqueda"] == nombre].iloc[0]
    valores = fila.to_dict()
else:
    st.sidebar.markdown("### 🔗 Lectura de URL YouTube")
    yt_url = st.sidebar.text_input("Pega el enlace de la canción:", "https://www.youtube.com/watch?v=5qap5aO4i9A")
    
    datos_coincidentes = None
    if yt_url:
        with st.sidebar.spinner("Leyendo información del video..."):
            yt_title, yt_artist, datos_coincidentes = obtener_metadatos_youtube(yt_url)
            if datos_coincidentes:
                st.sidebar.success(f"Encontrado en catálogo: **{datos_coincidentes.get('track_name')}**")
            else:
                st.sidebar.info(f"Video detectado: **{yt_title}** (Valores ajustables manualmente)")

    st.sidebar.markdown("### 🎚️ Atributos de Audio")
    def_val = datos_coincidentes if datos_coincidentes else {}

    valores["danceability"] = st.sidebar.slider("Danceability", 0.0, 1.0, float(def_val.get("danceability", 0.5)))
    valores["energy"] = st.sidebar.slider("Energy", 0.0, 1.0, float(def_val.get("energy", 0.5)))
    valores["valence"] = st.sidebar.slider("Valence (ánimo)", 0.0, 1.0, float(def_val.get("valence", 0.5)))
    valores["acousticness"] = st.sidebar.slider("Acousticness", 0.0, 1.0, float(def_val.get("acousticness", 0.2)))
    valores["instrumentalness"] = st.sidebar.slider("Instrumentalness", 0.0, 1.0, float(def_val.get("instrumentalness", 0.0)))
    valores["liveness"] = st.sidebar.slider("Liveness", 0.0, 1.0, float(def_val.get("liveness", 0.1)))
    valores["speechiness"] = st.sidebar.slider("Speechiness", 0.0, 1.0, float(def_val.get("speechiness", 0.05)))
    valores["loudness"] = st.sidebar.slider("Loudness (dB)", -40.0, 3.0, float(def_val.get("loudness", -8.0)))
    valores["tempo"] = st.sidebar.slider("Tempo (BPM)", 40.0, 220.0, float(def_val.get("tempo", 120.0)))
    valores["popularity"] = st.sidebar.slider("Popularidad", 0, 100, int(def_val.get("popularity", 50)))
    valores["duration_ms"] = st.sidebar.slider("Duración (ms)", 60000, 400000, int(def_val.get("duration_ms", 210000)))
    valores["explicit"] = st.sidebar.checkbox("Explicit", value=bool(def_val.get("explicit", False)))
    valores["mode"] = st.sidebar.selectbox("Mode", [0, 1], index=int(def_val.get("mode", 1)))
    valores["key"] = st.sidebar.selectbox("Key", list(range(12)), index=int(def_val.get("key", 0)))
    valores["time_signature"] = st.sidebar.selectbox("Time signature", [3, 4, 5], index=1)

st.sidebar.markdown("---")
st.sidebar.caption("Modelo: K-Means (k=5) · scikit-learn")

# ---------------------------------------------------------------------------
# Predicción
# ---------------------------------------------------------------------------
X_nueva = construir_vector(valores)
cluster_pred = int(kmeans.predict(X_nueva)[0])
punto_pca = pca.transform(X_nueva)[0]
titulo_perfil, desc_perfil, generos_perfil = PERFILES[cluster_pred]
accent = CLUSTER_COLORS[cluster_pred]

# ---------------------------------------------------------------------------
# Cuerpo Principal
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([1.2, 1], gap="medium")

with col_left:
    st.markdown('<div class="section-title">📊 Audio Features Clave</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_box("Energy", f"{valores.get('energy', 0):.2f}")
    with m2:
        metric_box("Danceability", f"{valores.get('danceability', 0):.2f}")
    with m3:
        metric_box("Valence", f"{valores.get('valence', 0):.2f}")
    with m4:
        metric_box("Tempo", f"{valores.get('tempo', 0):.0f} <span style='font-size:0.75rem;'>BPM</span>")

    st.markdown(
        f"""
        <div class="cluster-card" style="--accent:{accent};">
            <span class="badge" style="--accent:{accent};">CLUSTER {cluster_pred}</span>
            <h3>{titulo_perfil}</h3>
            <p>{desc_perfil}</p>
            <div class="genres"><b>Géneros afines:</b> {generos_perfil}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">🗺️ Posición en Espacio Latente</div>', unsafe_allow_html=True)
    st.pyplot(graficar_pca(punto_pca, cluster_pred))

with col_right:
    st.markdown('<div class="section-title">🎧 Reproductor de Audio</div>', unsafe_allow_html=True)

    if modo == "Catálogo preexistente":
        track_title = valores.get("track_name", "Desconocido")
        artist_name = valores.get("artists", "")

        st.markdown(
            f"""
            <div class="player-card">
                <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                    <span style="color:#1DB954; font-size:1.1rem;">▶</span>
                    <span style="color:#1DB954; font-weight:800; font-size:0.8rem; letter-spacing:1px; text-transform:uppercase;">Spotify Player</span>
                </div>
                <h4 style="margin:0; color:#FFFFFF; font-size:1.1rem; font-weight:700;">{track_title}</h4>
                <p style="margin:4px 0 0 0; color:#94A3B8; font-size:0.9rem;">{artist_name}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if "preview_url" in valores and pd.notna(valores["preview_url"]) and str(valores["preview_url"]).startswith("http"):
            st.audio(valores["preview_url"], format="audio/mp3")
        elif "track_id" in valores and pd.notna(valores["track_id"]):
            t_id = valores["track_id"]
            st.components.v1.html(
                f'<iframe src="https://open.spotify.com/embed/track/{t_id}?utm_source=generator" '
                f'width="100%" height="152" frameborder="0" allowfullscreen="" '
                f'allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" '
                f'loading="lazy"></iframe>',
                height=160,
            )
        else:
            query_str = f"{track_title} {artist_name}".replace(" ", "+")
            st.components.v1.html(
                f'<iframe width="100%" height="200" '
                f'src="https://www.youtube-nocookie.com/embed?listType=search&list={query_str}" '
                f'frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" '
                f'allowfullscreen></iframe>',
                height=210,
            )
    else:
        # ---- MODO INGRESO MANUAL / URL YOUTUBE ----
        video_id = None
        if "watch?v=" in yt_url:
            video_id = yt_url.split("watch?v=")[1].split("&")[0]
        elif "youtu.be/" in yt_url:
            video_id = yt_url.split("youtu.be/")[1].split("?")[0]

        if video_id:
            url_anclada = f"https://www.youtube.com/watch?v={video_id}"

            st.markdown(
                f"""
                <div class="player-card" style="border-color: rgba(255, 0, 0, 0.4);">
                    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">
                        <div style="display:flex; align-items:center; gap:8px;">
                            <span style="color:#FF0000; font-size:1.1rem;">▶</span>
                            <span style="color:#FF0000; font-weight:800; font-size:0.8rem; letter-spacing:1px; text-transform:uppercase;">YouTube Player</span>
                        </div>
                        <a href="{url_anclada}" target="_blank" style="color:#1DB954; text-decoration:none; font-size:0.82rem; font-weight:700;">
                            🔗 Abrir en YouTube ↗
                        </a>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.components.v1.html(
                f'<iframe width="100%" height="200" '
                f'src="https://www.youtube-nocookie.com/embed/{video_id}" '
                f'frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" '
                f'allowfullscreen></iframe>',
                height=210,
            )
        else:
            st.warning("⚠️ Ingresa un enlace válido de YouTube (ej. https://www.youtube.com/watch?v=...)")

    st.markdown('<div class="section-title">🎶 Recomendaciones del Cluster</div>', unsafe_allow_html=True)
    similares = (
        df[df["cluster_kmeans"] == cluster_pred][["track_name", "artists", "track_genre"]]
        .dropna()
        .sample(min(6, len(df)))
    )
    st.dataframe(similares, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("UNC — Facultad de Ingeniería — Proyecto Final de Sistemas Inteligentes · Modelo K-Means (k=5)")