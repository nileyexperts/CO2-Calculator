import re
from pathlib import Path
import shutil

APP = Path("app.py")
BAK = Path("app.py.bak")

src = APP.read_text(encoding="utf-8")

# 1) Modifier la signature de unified_location_input
src = re.sub(
    r"def unified_location_input\(\s*side_key: str,\s*seg_index: int,\s*label_prefix: str\)\s*:",
    r'def unified_location_input(side_key: str, seg_index: int, label_prefix: str, show_airports: bool = True):',
    src,
    count=1,
)

# 2) Adapter le label du text_input (ajout condition IATA)
src = re.sub(
    r'f"\{label_prefix\} — Adresse / Ville / Pays ou IATA \(3 lettres\)"',
    r'f"{label_prefix} — Adresse / Ville / Pays" + (" ou IATA (3 lettres)" if show_airports else "")',
    src,
    count=1,
)

# 3) Ne chercher les aéroports que si show_airports=True
src = re.sub(
    r"if query_val:\s*\n\s*airports = search_airports\(query_val, limit=10\)\s*\n\s*oc = geocode_cached\(query_val, limit=5\)",
    (
        "if query_val:\n"
        "        # Ne chercher les aéroports que si autorisé\n"
        "        if show_airports:\n"
        "            airports = search_airports(query_val, limit=10)\n"
        "        oc = geocode_cached(query_val, limit=5)"
    ),
    src,
    count=1,
)

# 4) Injecter les résultats IATA seulement si show_airports=True
src = src.replace(
    "if not airports.empty:",
    "if show_airports and not airports.empty:",
    1
)

# 5) Ajouter le commentaire/gestion du cas aéroport (optionnel si déjà OK)
# (le code existant reconnaît déjà '✈️', on le laisse)

# 6) Modifier les appels dans la boucle des segments : Origine/Destination
src = src.replace(
    'with c1:\n            st.markdown("**Origine**")\n            o = unified_location_input("origin", i, "Origine")',
    (
        'with c1:\n'
        '            st.markdown("**Origine**")\n'
        '            # Proposer les aéroports UNIQUEMENT si mode Aérien pour l\'Origine\n'
        '            o = unified_location_input(\n'
        '                "origin", i, "Origine",\n'
        '                show_airports=("aerien" in _normalize_no_diacritics(mode))\n'
        '            )'
    ),
    1
)

src = src.replace(
    'with c2:\n            st.markdown("**Destination**")\n            d = unified_location_input("dest", i, "Destination")',
    (
        'with c2:\n'
        '            st.markdown("**Destination**")\n'
        '            # Jamais d\'aéroports pour la Destination (selon votre demande)\n'
        '            d = unified_location_input(\n'
        '                "dest", i, "Destination",\n'
        '                show_airports=False\n'
        '            )'
    ),
    1
)

# Sauvegarde
shutil.copyfile(APP, BAK)
APP.write_text(src, encoding="utf-8")
print("✅ app.py mis à jour. Un backup a été créé sous app.py.bak.")
``
