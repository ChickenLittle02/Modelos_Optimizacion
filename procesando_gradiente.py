import pandas as pd

def region_de(x):
    if x > 2:
        return "x > 2"
    elif x < 0:
        return "x < 0"
    else:
        return "0 < x < 2"

def analizar_csv_gradiente(path_csv):
    df = pd.read_csv(path_csv)

    # Convertir x0 de lista/string → a números
    df["x0"] = df["x0"].apply(lambda v: float(str(v).strip("[]").split(",")[0]))
    df["y0"] = df["x0"]  # tu gradiente siempre forzó y=0 en muchos casos

    # Crear columna región
    df["region"] = df["x0"].apply(region_de)

    regiones = df["region"].unique()

    resumen_md = []
    resumen_md.append("# Análisis de Learning Rates por Región\n")

    for region in regiones:
        sub = df[df["region"] == region]

        resumen_md.append(f"## Región {region}\n")

        # Mejor f_final
        best_f = sub["f_final"].min()
        filas_best_f = sub[sub["f_final"] == best_f]

        resumen_md.append(f"**Valor mínimo alcanzado en la región:** {best_f:.6f}\n")

        for _, row in filas_best_f.iterrows():
            resumen_md.append(
                f"- Punto inicial {row['x0']} con lr={row['lr']} → Iteraciones: {row['num_iter']}\n"
            )

        # Mejor lr por rapidez
        resumen_md.append("\n### Comparación de velocidades\n")
        for (x0), grp in sub.groupby(["x0"]):

            fast = grp.loc[grp["num_iter"].idxmin()]
            slow = grp.loc[grp["num_iter"].idxmax()]

            resumen_md.append(
                f"- Punto inicial {x0}:\n"
                f"   - Más rápido lr={fast['lr']} → {fast['num_iter']} iteraciones\n"
                f"   - Más lento lr={slow['lr']} → {slow['num_iter']} iteraciones\n"
            )

        # Sugerencia final por región
        lr_recomendado = (
            sub.groupby("lr")["num_iter"].mean().sort_values().index[0]
        )

        resumen_md.append(
            f"\n### ⭐ Learning Rate recomendado para región {region}: **{lr_recomendado}**\n"
        )

        resumen_md.append("\n---\n")

    # Guardar el archivo .md
    with open("analisis_lr_regiones.md", "w", encoding="utf-8") as f:
        f.write("\n".join(resumen_md))


    print("✔ Archivo generado: analisis_lr_regiones.md")

analizar_csv_gradiente("resultados_gradiente.csv")
