"""
Module de génération de rapport PDF pour le calculateur CO2
À intégrer dans app.py
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.pdfgen import canvas
from io import BytesIO
from datetime import datetime
import folium
from PIL import Image as PILImage
import io

def generate_pdf_report(df, dossier_val, total_distance, total_emissions, unit, rows):
    """
    Génère un rapport PDF au format A4 paysage
    
    Args:
        df: DataFrame avec les résultats
        dossier_val: Numéro de dossier transport
        total_distance: Distance totale en km
        total_emissions: Émissions totales en kg CO2e
        unit: Unité de poids (kg ou tonnes)
        rows: Liste des lignes de calcul avec coordonnées
    
    Returns:
        BytesIO: Buffer contenant le PDF
    """
    buffer = BytesIO()
    
    # Format A4 paysage
    page_width, page_height = landscape(A4)
    
    # Création du document
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=1.5*cm,
        leftMargin=1.5*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=12,
        alignment=1  # Centré
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2c5aa0'),
        spaceAfter=10,
        spaceBefore=10
    )
    
    normal_style = styles['Normal']
    normal_style.fontSize = 10
    
    # Contenu du document
    story = []
    
    # En-tête avec logo (optionnel - vous pouvez télécharger le logo)
    try:
        logo_url = "https://raw.githubusercontent.com/nileyexperts/CO2-Calculator/main/NILEY-EXPERTS-logo-removebg-preview.png"
        import requests
        response = requests.get(logo_url, timeout=5)
        if response.status_code == 200:
            logo_img = PILImage.open(io.BytesIO(response.content))
            logo_buffer = io.BytesIO()
            logo_img.save(logo_buffer, format='PNG')
            logo_buffer.seek(0)
            logo = Image(logo_buffer, width=4*cm, height=2*cm)
            story.append(logo)
    except:
        pass  # Si le logo ne peut pas être chargé, on continue sans
    
    # Titre
    story.append(Paragraph("RAPPORT D'EMPREINTE CARBONE MULTIMODAL", title_style))
    story.append(Spacer(1, 0.5*cm))
    
    # Informations générales
    story.append(Paragraph("Informations générales", heading_style))
    
    info_data = [
        ["N° dossier Transport:", dossier_val],
        ["Date du rapport:", datetime.now().strftime("%d/%m/%Y %H:%M")],
        ["Nombre de segments:", str(len(rows))],
    ]
    
    info_table = Table(info_data, colWidths=[6*cm, 10*cm])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f0f7')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.5*cm))
    
    # Résumé des émissions
    story.append(Paragraph("Résumé des émissions", heading_style))
    
    summary_data = [
        ["Distance totale:", f"{total_distance:.1f} km"],
        ["Émissions totales:", f"{total_emissions:.2f} kg CO₂e"],
        ["Émissions moyennes:", f"{total_emissions/total_distance:.3f} kg CO₂e/km" if total_distance > 0 else "N/A"],
    ]
    
    summary_table = Table(summary_data, colWidths=[6*cm, 10*cm])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#fff4e6')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.8*cm))
    
    # Carte simplifiée
    story.append(Paragraph("Carte des trajets", heading_style))
    
    # Création d'une carte Folium simplifiée
    if rows:
        # Calcul du centre de la carte
        all_lats = [r["lat_o"] for r in rows] + [r["lat_d"] for r in rows]
        all_lons = [r["lon_o"] for r in rows] + [r["lon_d"] for r in rows]
        center_lat = sum(all_lats) / len(all_lats)
        center_lon = sum(all_lons) / len(all_lons)
        
        # Création de la carte
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=5,
            tiles='CartoDB positron',
            width=800,
            height=300
        )
        
        # Ajout des marqueurs et lignes
        colors_map = {
            "routier": "blue",
            "aerien": "red",
            "maritime": "green",
            "ferroviaire": "purple"
        }
        
        for r in rows:
            # Déterminer la couleur selon le mode
            mode_lower = r["Mode"].lower()
            color = "gray"
            for key, col in colors_map.items():
                if key in mode_lower:
                    color = col
                    break
            
            # Ligne entre origine et destination
            folium.PolyLine(
                locations=[[r["lat_o"], r["lon_o"]], [r["lat_d"], r["lat_d"]]],
                color=color,
                weight=3,
                opacity=0.7,
                popup=f"Segment {r['Segment']}: {r['Mode']}"
            ).add_to(m)
            
            # Marqueur origine
            folium.CircleMarker(
                location=[r["lat_o"], r["lon_o"]],
                radius=6,
                color='darkblue',
                fill=True,
                fillColor='lightblue',
                fillOpacity=0.8,
                popup=f"S{r['Segment']} - Origine"
            ).add_to(m)
            
            # Marqueur destination
            folium.CircleMarker(
                location=[r["lat_d"], r["lon_d"]],
                radius=6,
                color='darkred',
                fill=True,
                fillColor='red',
                fillOpacity=0.8,
                popup=f"S{r['Segment']} - Destination"
            ).add_to(m)
        
        # Sauvegarder la carte en image
        try:
            map_buffer = io.BytesIO()
            m.save(map_buffer, close_file=False)
            map_buffer.seek(0)
            
            # Convertir en image avec selenium (nécessite geckodriver ou chromedriver)
            # Alternative: utiliser la capture d'écran de la carte
            # Pour simplifier, on saute la carte si trop complexe
            # story.append(Image(map_buffer, width=24*cm, height=9*cm))
            story.append(Paragraph("Carte interactive disponible dans l'application web", normal_style))
        except:
            story.append(Paragraph("Carte non disponible dans ce rapport", normal_style))
    
    story.append(Spacer(1, 0.5*cm))
    
    # Tableau détaillé des segments
    story.append(Paragraph("Détail des segments", heading_style))
    
    # Préparation des données du tableau
    table_data = [["Seg.", "Origine", "Destination", "Mode", "Distance\n(km)", 
                   f"Poids\n({unit})", "Facteur\n(kg CO₂e/t.km)", "Émissions\n(kg CO₂e)"]]
    
    for _, row in df.iterrows():
        table_data.append([
            str(row["Segment"]),
            row["Origine"][:30] + "..." if len(row["Origine"]) > 30 else row["Origine"],
            row["Destination"][:30] + "..." if len(row["Destination"]) > 30 else row["Destination"],
            row["Mode"].replace("🚛", "").replace("✈️", "").replace("🛢️", "").replace("🚂", "").strip(),
            f"{row['Distance (km)']:.1f}",
            f"{row[f'Poids ({unit})']:.1f}",
            f"{row['Facteur (kg CO₂e/t.km)']:.3f}",
            f"{row['Émissions (kg CO₂e)']:.2f}"
        ])
    
    # Ligne de total
    table_data.append([
        "TOTAL",
        "",
        "",
        "",
        f"{total_distance:.1f}",
        "",
        "",
        f"{total_emissions:.2f}"
    ])
    
    # Création du tableau
    col_widths = [1.5*cm, 5*cm, 5*cm, 3.5*cm, 2*cm, 2*cm, 2.5*cm, 2.5*cm]
    
    detail_table = Table(table_data, colWidths=col_widths, repeatRows=1)
    detail_table.setStyle(TableStyle([
        # En-tête
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 8),
        
        # Corps du tableau
        ('BACKGROUND', (0, 1), (-1, -2), colors.white),
        ('TEXTCOLOR', (0, 1), (-1, -2), colors.black),
        ('ALIGN', (0, 1), (0, -1), 'CENTER'),
        ('ALIGN', (4, 1), (-1, -1), 'RIGHT'),
        ('FONTNAME', (0, 1), (-1, -2), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -2), 8),
        ('VALIGN', (0, 1), (-1, -1), 'MIDDLE'),
        
        # Ligne de total
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#fff4e6')),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, -1), (-1, -1), 9),
        
        # Grille
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('LINEBELOW', (0, 0), (-1, 0), 2, colors.HexColor('#1f4788')),
        ('LINEABOVE', (0, -1), (-1, -1), 2, colors.HexColor('#1f4788')),
        
        # Padding
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]))
    
    story.append(detail_table)
    story.append(Spacer(1, 1*cm))
    
    # Pied de page
    story.append(Paragraph(
        f"<i>Document généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')} - NILEY EXPERTS</i>",
        ParagraphStyle('Footer', parent=normal_style, fontSize=8, textColor=colors.grey, alignment=1)
    ))
    
    # Construction du PDF
    doc.build(story)
    buffer.seek(0)
    
    return buffer


# Code à ajouter dans app.py après le bouton de téléchargement CSV :
"""
# Génération du rapport PDF
try:
    pdf_buffer = generate_pdf_report(
        df=df,
        dossier_val=dossier_val,
        total_distance=total_distance,
        total_emissions=total_emissions,
        unit=unit,
        rows=rows
    )
    
    filename_pdf = f"rapport_co2_multimodal{safe_suffix}.pdf"
    
    st.download_button(
        "📄 Télécharger le rapport PDF",
        data=pdf_buffer,
        file_name=filename_pdf,
        mime="application/pdf"
    )
except Exception as e:
    st.error(f"Erreur lors de la génération du PDF : {e}")
"""
