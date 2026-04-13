import pandas as pd
import numpy as np
from openap.fuel import FuelFlow

def pipeline_limpieza_y_combustible(input_path, output_path):
    print("1. Cargando datos...")
    df = pd.read_csv(input_path)
    
    # Crear ID único y ordenar
    df['flight_id'] = df['icao24'].astype(str) + "_" + df['callsign'].astype(str)
    df = df.sort_values(by=['flight_id', 'time'])

    # Inicializar modelo OpenAP para A320
    fuelflow_model = FuelFlow(ac='A320')
    MASA_AVION = 65000 

    cleaned_data = []

    print("2. Procesando trayectorias (suavizado + física)...")
    for fid, flight in df.groupby('flight_id'):
        f = flight.copy()

        # --- PASO A: SUAVIZADO DE ENTRADAS ---
        # Suavizamos altitud y velocidad ANTES de calcular el combustible
        # Esto elimina los picos que vuelven loco al modelo físico
        f['baro_altitude'] = f['baro_altitude'].rolling(window=20, min_periods=1, center=True).mean()
        f['vel_mps'] = f['vel_mps'].rolling(window=20, min_periods=1, center=True).mean()

        # --- PASO B: RECALCULAR VRATE LIMPIO ---
        f['dt'] = f['time'].diff().fillna(method='bfill')
        f['dalt'] = f['baro_altitude'].diff().fillna(0)
        f['vrate_mps_clean'] = (f['dalt'] / f['dt']).clip(-25, 25)

        # --- PASO C: CALCULAR FUEL FLOW CON DATOS LIMPIOS ---
        def calc_fuel(row):
            try:
                return fuelflow_model.enroute(
                    mass = MASA_AVION,
                    tas  = row['vel_mps'] * 1.94384, # m/s a knots
                    alt  = row['baro_altitude'] * 3.28084, # m a feet
                    vs   = row['vrate_mps_clean'] * 196.85 # m/s a ft/min
                )
            except:
                return 0.5 # Valor de seguridad (crucero aprox)

        f['fuel_flow_kgs'] = f.apply(calc_fuel, axis=1)

        # --- PASO D: RECALCULAR QUEMADO ACUMULADO (Opcional pero útil) ---
        f['fuel_burnt_kg'] = (f['fuel_flow_kgs'] * f['dt']).cumsum()

        # Limpiar columnas auxiliares
        f = f.drop(columns=['dt', 'dalt', 'vrate_mps_clean'])
        cleaned_data.append(f)

    # Unir todo
    df_final = pd.concat(cleaned_data)
    
    # Guardar
    df_final.to_csv(output_path, index=False)
    print(f"3. ¡Listo! Dataset final guardado en: {output_path}")

# Ejecutar
pipeline_limpieza_y_combustible('/home/edgar/GitHub/StochasticATM/Data/Datasets/dataset_total_unido.csv', 'dataset_listo_para_SDE.csv')
