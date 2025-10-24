# 🔐 Authentification simple par mot de passe
# =========================
APP_PASSWORD = "Niley2019!"  # ⚠️ En prod: préférez st.secrets / variable d'environnement

def check_password():
    if "auth_ok" not in st.session_state:
        st.session_state.auth_ok = False
    if st.session_state.auth_ok:
        return True

    st.title("🔐 Accès protégé")
    st.write("Veuillez saisir le mot de passe pour accéder à l’application.")
    with st.form("password_form", clear_on_submit=False):
        pwd = st.text_input("Mot de passe", type="password")
        submitted = st.form_submit_button("Se connecter")

    if submitted:
        if pwd == APP_PASSWORD:
            st.session_state.auth_ok = True
            st.success("Connexion réussie. Chargement de l’application…")
            st.rerun()
        else:
            st.error("Mot de passe incorrect. Réessayez.")
    return False

# Bloquer l’app tant que non authentifié
if not check_password():
    st.stop()
