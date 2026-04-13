import pandas as pd
import numpy as np
from openap.fuel import FuelFlow # Importación según el help

def corregir_dataset_con_openap_v2(archivo_entrada, archivo_salida):
    print(f"Procesando {archivo_entrada}...")
    df = pd.read_csv(archivo_entrada)

    # 1. Inicializar el modelo con el nombre de clase correcto: FuelFlow
    try:
        # ac='A320' es el código ICAO requerido
        fuelflow_model = FuelFlow(ac='A320')
        print("Modelo FuelFlow (A320) cargado.")
    except Exception as e:
        print(f"Error al inicializar FuelFlow: {e}")
        return

    # 2. Definir Masa de referencia
    # Como el modelo ahora PIDE 'mass', usaremos la masa típica de un A320 
    # El valor máximo de despegue (MTOW) es ~78000kg, usaremos una media de 65000kg.
    MASA_AVION = 65000 

    df = df.sort_values(by=['icao24', 'time'])

    def calcular_tasa_cientifica(row):
        try:
            # Según el help, el orden de enroute es: (mass, tas, alt, vs, ...)
            # tas: knots, alt: feet, vs: ft/min
            flow = fuelflow_model.enroute(
                mass = MASA_AVION,
                tas  = row['vel_mps'] * 1.94384, 
                alt  = row['baro_altitude'] * 3.28084,
                vs   = row['vrate_mps'] * 196.85
            )
            return flow
        except:
            return 0.015

    print("Calculando fuel_flow_kgs con parámetros de masa...")
    df['fuel_flow_kgs'] = df.apply(calcular_tasa_cientifica, axis=1)

    # 3. Recalcular quemado
    df['dt'] = df.groupby('icao24')['time'].diff().fillna(0)
    df['fuel_burnt_kg'] = df.groupby('icao24').apply(
        lambda x: (x['fuel_flow_kgs'] * x['dt']).cumsum()
    ).reset_index(level=0, drop=True)
    # Eliminar registros donde la velocidad es físicamente imposible para un comercial
    df = df[df['vel_mps'] < 300] 
    # Eliminar registros con saltos de tiempo demasiado grandes (perdió señal)
    df = df[df['dt'] < 60]
    # Guardar
    df.drop(columns=['dt'], inplace=True)
    df.to_csv(archivo_salida, index=False)
    print(f"Dataset corregido guardado en: {archivo_salida}")

# Ejecución
corregir_dataset_con_openap_v2("/home/edgar/GitHub/StochasticATM/Data/dataset_trayectorias_completas_nueva.csv", "Data/dataset_reparado_v2.csv")