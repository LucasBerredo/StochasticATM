import json
import requests
import pandas as pd
import numpy as np
import time
import os
from openap import prop
from datetime import datetime, timezone

# --- 1. CONFIGURACIÓN DE VOLUMEN MÁXIMO ---
MAX_AVIONES = 3000        # Consumirá un máximo de créditos igual a esta cantidad
LOTE_GUARDADO = 50        # Guardar en CSV cada 50 aviones (Checkpoint)
ARCHIVO_SALIDA = "dataset_atm_massive.csv"
PAUSA_BASE = 1.5          # Segundos entre peticiones para evitar el Error 429

# --- 2. CREDENCIALES ---
try:
    with open('credentials.json') as f:
        creds = json.load(f)
    AUTH = (creds['clientId'], creds['clientSecret'])
except FileNotFoundError:
    print("Error: No se encuentra el archivo 'credentials.json'.")
    exit()

# --- 3. MOTOR FÍSICO (Para Ecuaciones Diferenciales) ---
def calculate_physics_and_fuel(df):
    """
    Calcula las derivadas (velocidad y tasa vertical) y estima la masa consumida.
    Devuelve las unidades en el Sistema Internacional (metros, segundos, kg).
    """
    dt = df['time'].diff()
    R = 6371000.0 # Radio de la Tierra en metros
    
    # Haversine Vectorizado para distancia horizontal
    lat1, lon1 = np.radians(df['latitude'].shift()), np.radians(df['longitude'].shift())
    lat2, lon2 = np.radians(df['latitude']), np.radians(df['longitude'])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    dist_m = R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    # Derivadas respecto al tiempo (dt)
    df['vel_mps'] = dist_m / dt
    df['vrate_mps'] = df['baro_altitude'].diff() / dt
    df.fillna({'vel_mps': 0, 'vrate_mps': 0, 'baro_altitude': 0}, inplace=True)
    
    # Estimación de Combustible (Física)
    try:
        fuel_model = prop.Fuel(ac='A320') # Modelo base genérico
        # OpenAP usa unidades imperiales por defecto: alt(ft), vel(kt), vrate(fpm)
        df['fuel_flow_kgs'] = df.apply(
            lambda r: fuel_model.enroute(
                alt = r['baro_altitude'] * 3.28084, 
                vtas = r['vel_mps'] * 1.94384, 
                vs = r['vrate_mps'] * 196.85
            ), axis=1
        )
    except Exception:
        df['fuel_flow_kgs'] = 0.015 # Fallback si falla el cálculo
        
    # Integral para obtener la masa total quemada
    df['fuel_burnt_kg'] = df['fuel_flow_kgs'].cumsum()
    return df

# --- 4. EXTRACCIÓN MASIVA CONTINUA ---
def generate_massive_dataset():
    print(f"--- INICIANDO GENERADOR MASIVO (Objetivo: {MAX_AVIONES} vuelos) ---")
    
    now_utc = datetime.now(timezone.utc)
    print(f"Paso 1: Mapeando el cielo global EN VIVO ({now_utc.strftime('%Y-%m-%d %H:%M UTC')})...")
    
    url_states = "https://opensky-network.org/api/states/all"
    
    try:
        r_states = requests.get(url_states, auth=AUTH, timeout=20)
    except requests.exceptions.RequestException as e:
        print(f"Error de red crítico al contactar OpenSky: {e}")
        return

    if r_states.status_code != 200:
        print(f"Error de la API: Código HTTP {r_states.status_code}")
        print(f"Mensaje del servidor: {r_states.text}")
        return
        
    states = r_states.json().get('states', [])
    
    # Filtro: Solo aviones que están en el aire (índice 8 es 'on_ground')
    aviones_en_aire = [s for s in states if s[8] is False]
    print(f"Total de aviones detectados en el aire globalmente: {len(aviones_en_aire)}")
    
    if len(aviones_en_aire) == 0:
        print("No hay aviones detectados. Abortando.")
        return

    # Limitamos a nuestro objetivo
    aviones_a_procesar = aviones_en_aire[:MAX_AVIONES]
    total_objetivo = len(aviones_a_procesar)
    
    print(f"\nPaso 2: Descargando trayectorias (Consumo estimado: {total_objetivo} créditos)")
    print(f"Los datos se guardarán en '{ARCHIVO_SALIDA}' cada {LOTE_GUARDADO} vuelos.\n")
    
    lote_actual = []
    vuelos_exitosos = 0
    
    for i, state in enumerate(aviones_a_procesar, 1):
        icao24 = state[0]
        callsign = str(state[1]).strip() if state[1] else f"UNK_{icao24}"
        
        try:
            # time=0 pide la trayectoria del vuelo ACTIVO actual
            r_track = requests.get("https://opensky-network.org/api/tracks/all", 
                                   params={"icao24": icao24, "time": 0}, 
                                   auth=AUTH, timeout=10)
            
            if r_track.status_code == 200:
                track_data = r_track.json()
                if 'path' in track_data:
                    df = pd.DataFrame(track_data['path'], 
                                      columns=['time', 'latitude', 'longitude', 'baro_altitude', 'track', 'on_ground'])
                    
                    if len(df) > 50:
                        df['icao24'] = icao24
                        df['callsign'] = callsign
                        df = calculate_physics_and_fuel(df)
                        lote_actual.append(df)
                        vuelos_exitosos += 1
                        print(f"[{i}/{total_objetivo}] Éxito: {callsign} ({len(df)} puntos)")
                    else:
                        print(f"[{i}/{total_objetivo}] Omitido: {callsign} (Trayectoria muy corta)")
                        
            elif r_track.status_code == 429:
                 print(f"[{i}/{total_objetivo}] ERROR 429: Límite de velocidad alcanzado.")
                 print("El servidor nos pide un respiro. Pausando extracción por 60 segundos...")
                 time.sleep(60)
                 continue # Reanudamos el bucle tras la pausa
                 
            elif r_track.status_code == 404:
                 print(f"[{i}/{total_objetivo}] Fallo API (HTTP 404): No hay ruta para {icao24}")
            else:
                 print(f"[{i}/{total_objetivo}] Fallo API (HTTP {r_track.status_code})")
                 
        except requests.exceptions.RequestException as e:
            print(f"[{i}/{total_objetivo}] Error de conexión con {icao24}. Reintentando en breve...")
            time.sleep(2)
            
        # GUARDADO POR LOTES (CHECKPOINT)
        if len(lote_actual) >= LOTE_GUARDADO or i == total_objetivo:
            if lote_actual:
                df_lote = pd.concat(lote_actual, ignore_index=True)
                cols = ['time', 'icao24', 'callsign', 'latitude', 'longitude', 'baro_altitude', 
                        'vel_mps', 'vrate_mps', 'track', 'fuel_flow_kgs', 'fuel_burnt_kg']
                
                # Si el archivo no existe, lo crea con cabeceras. Si ya existe, añade filas.
                header = not os.path.exists(ARCHIVO_SALIDA)
                df_lote[cols].to_csv(ARCHIVO_SALIDA, mode='a', header=header, index=False)
                
                print(f"\n >>> CHECKPOINT GUARDADO: {vuelos_exitosos} vuelos totales en '{ARCHIVO_SALIDA}' <<<\n")
                lote_actual = [] # Limpiar memoria
        
        # Pausa técnica para evitar en lo posible el error 429
        time.sleep(PAUSA_BASE)

    print(f"\n--- EXTRACCIÓN MASIVA COMPLETADA ---")
    print(f"Dataset final guardado en: {ARCHIVO_SALIDA}")
    print(f"Vuelos procesados con éxito listos para la Red Neuronal: {vuelos_exitosos}/{total_objetivo}")

if __name__ == "__main__":
    generate_massive_dataset()