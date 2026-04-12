import json
import requests
import pandas as pd
import numpy as np
import time
import os
from openap import prop
from datetime import datetime, timedelta, timezone

# --- 1. CONFIGURACIÓN ---
AEROPUERTOS = ["LEMD", "LEBL", "EGLL", "LFPG", "EDDF"] # Madrid, BCN, Heathrow, París, Frankfurt
ARCHIVO_SALIDA = "Data/dataset_trayectorias_completas.csv"
MAX_INTENTOS = 2  # Número máximo de veces que intentará descargar un avión si hay error

# --- 2. GESTOR DE TOKENS OAUTH2 ---
TOKEN_URL = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
TOKEN_REFRESH_MARGIN = 30

class TokenManager:
    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token = None
        self.expires_at = None

    def get_token(self):
        if self.token and self.expires_at and datetime.now() < self.expires_at:
            return self.token
        return self._refresh()

    def _refresh(self):
        r = requests.post(
            TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
        )
        r.raise_for_status()
        data = r.json()
        self.token = data["access_token"]
        expires_in = data.get("expires_in", 1800)
        self.expires_at = datetime.now() + timedelta(seconds=expires_in - TOKEN_REFRESH_MARGIN)
        return self.token

    def headers(self):
        return {"Authorization": f"Bearer {self.get_token()}"}

# Cargar credenciales
try:
    with open('credentials.json') as f:
        creds = json.load(f)
    auth_manager = TokenManager(creds['clientId'], creds['clientSecret'])
except Exception as e:
    print(f"Error cargando credentials.json: {e}")
    exit()

# --- 3. MOTOR FÍSICO ---
def calculate_physics_and_fuel(df):
    dt = df['time'].diff()
    R = 6371000.0
    
    lat1, lon1 = np.radians(df['latitude'].shift()), np.radians(df['longitude'].shift())
    lat2, lon2 = np.radians(df['latitude']), np.radians(df['longitude'])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    dist_m = R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    df['vel_mps'] = dist_m / dt
    df['vrate_mps'] = df['baro_altitude'].diff() / dt
    df.fillna({'vel_mps': 0, 'vrate_mps': 0, 'baro_altitude': 0}, inplace=True)
    
    try:
        fuel_model = prop.Fuel(ac='A320')
        df['fuel_flow_kgs'] = df.apply(
            lambda r: fuel_model.enroute(
                alt = r['baro_altitude'] * 3.28084, 
                vtas = r['vel_mps'] * 1.94384, 
                vs = r['vrate_mps'] * 196.85
            ), axis=1
        )
    except Exception:
        df['fuel_flow_kgs'] = 0.015
    
    df['fuel_burnt_kg'] = df['fuel_flow_kgs'].cumsum()
    return df

# --- 4. EXTRACCIÓN HISTÓRICA POR AEROPUERTO ---
def generar_dataset_oficial():
    print("--- INICIANDO EXTRACCIÓN OFICIAL DE LLEGADAS ---")
    
    # Buscamos vuelos entre las 10:00 y las 14:00 UTC de AYER.
    end_dt = datetime.now(timezone.utc).replace(hour=14, minute=0, second=0, microsecond=0) - timedelta(days=1)
    begin_dt = end_dt - timedelta(hours=4)
    
    end_ts = int(end_dt.timestamp())
    begin_ts = int(begin_dt.timestamp())
    
    print(f"Ventana de búsqueda: {begin_dt.strftime('%Y-%m-%d %H:%M')} a {end_dt.strftime('%Y-%m-%d %H:%M')} UTC\n")
    print("PULSA 'Ctrl + C' EN CUALQUIER MOMENTO PARA DETENER Y GUARDAR.\n")
    
    vuelos_procesados = 0
    lote_actual = []
    
    try:
        for aeropuerto in AEROPUERTOS:
            print(f"✈️ Consultando llegadas históricas a: {aeropuerto}...")
            url_arrivals = "https://opensky-network.org/api/flights/arrival"
            params = {"airport": aeropuerto, "begin": begin_ts, "end": end_ts}
            
            r_arr = requests.get(url_arrivals, params=params, headers=auth_manager.headers())
            
            if r_arr.status_code == 200:
                llegadas = r_arr.json()
                print(f"   Encontrados {len(llegadas)} vuelos completados.")
                
                for i, vuelo in enumerate(llegadas, 1):
                    icao24 = vuelo['icao24']
                    callsign = str(vuelo.get('callsign', '')).strip() or f"UNK_{icao24}"
                    
                    # BUCLE DE REINTENTOS PARA CADA AVIÓN
                    for intento in range(1, MAX_INTENTOS + 1):
                        try:
                            r_track = requests.get("https://opensky-network.org/api/tracks/all", 
                                                   params={"icao24": icao24, "time": vuelo['lastSeen']}, 
                                                   headers=auth_manager.headers(),
                                                   timeout=10)
                            
                            if r_track.status_code == 200:
                                t_data = r_track.json()
                                if 'path' in t_data:
                                    df = pd.DataFrame(t_data['path'], columns=['time', 'latitude', 'longitude', 'baro_altitude', 'track', 'on_ground'])
                                    
                                    if len(df) > 200:
                                        df['icao24'] = icao24
                                        df['callsign'] = callsign
                                        df['destino'] = aeropuerto
                                        df['origen'] = vuelo.get('estDepartureAirport', 'UNK')
                                        
                                        df = calculate_physics_and_fuel(df)
                                        lote_actual.append(df)
                                        vuelos_procesados += 1
                                        print(f"   [{i}/{len(llegadas)}] OK: {callsign} ({len(df)} pts)")
                                    else:
                                        print(f"   [{i}/{len(llegadas)}] Omitido: Trayectoria muy corta")
                                
                                # Si fue exitoso, rompemos el bucle de reintentos y pasamos al siguiente avión
                                break 
                                
                            elif r_track.status_code == 429:
                                # Leemos cuánto tiempo exige el servidor que esperemos
                                wait_time = int(r_track.headers.get('X-Rate-Limit-Retry-After-Seconds', 60))
                                
                                if wait_time > 300: # Si pide esperar más de 5 minutos, es que se acabó el saldo diario
                                    horas_espera = wait_time / 3600
                                    print(f"\n   [!!!] LÍMITE DIARIO AGOTADO [!!!]")
                                    print(f"   El servidor exige una espera de {horas_espera:.1f} horas para recargar créditos.")
                                    print("   Forzando guardado de emergencia y cerrando el script...")
                                    raise KeyboardInterrupt # Simulamos que tú pulsaste Ctrl+C para guardar y salir limpio
                                
                                elif intento < MAX_INTENTOS:
                                    print(f"   [!] Error 429 temporal. Reintento {intento}/{MAX_INTENTOS} en {wait_time}s...")
                                    time.sleep(wait_time)
                                else:
                                    print(f"   [{i}/{len(llegadas)}] Descartado: Límite de API tras {MAX_INTENTOS} intentos.")
                            else:
                                # Otros errores (404, 400), no vale la pena reintentar
                                print(f"   [{i}/{len(llegadas)}] Error HTTP {r_track.status_code}. Omitido.")
                                break
                                
                        except requests.exceptions.RequestException as e:
                            if intento < MAX_INTENTOS:
                                print(f"   [!] Error de red. Reintento {intento}/{MAX_INTENTOS} en 5s...")
                                time.sleep(5)
                            else:
                                print(f"   [{i}/{len(llegadas)}] Descartado tras {MAX_INTENTOS} fallos de conexión.")
                    
                    time.sleep(1.0) # Pausa base tras procesar cada avión
                    
            elif r_arr.status_code == 404:
                print(f"   No se registraron llegadas para {aeropuerto} en este horario.")
            else:
                print(f"   Error HTTP {r_arr.status_code}: {r_arr.text[:100]}")
                
            # Guardado normal al terminar cada aeropuerto
            if lote_actual:
                df_final = pd.concat(lote_actual, ignore_index=True)
                cols = ['time', 'icao24', 'callsign', 'origen', 'destino', 'latitude', 'longitude', 'baro_altitude', 'vel_mps', 'vrate_mps', 'track', 'fuel_flow_kgs', 'fuel_burnt_kg']
                header = not os.path.exists(ARCHIVO_SALIDA)
                df_final[cols].to_csv(ARCHIVO_SALIDA, mode='a', header=header, index=False)
                print(f"\n>>> CHECKPOINT: {vuelos_procesados} vuelos en total guardados en {ARCHIVO_SALIDA} <<<\n")
                lote_actual = []
                
            time.sleep(3)

    except KeyboardInterrupt:
        print("\n\n[!] EXTRACCIÓN INTERRUMPIDA POR EL USUARIO (Ctrl+C).")
        
    finally:
        # Guardado de emergencia (se ejecuta siempre al salir)
        if lote_actual:
            print(f"\n>>> GUARDADO DE EMERGENCIA: Salvando {len(lote_actual)} vuelos residuales... <<<")
            df_final = pd.concat(lote_actual, ignore_index=True)
            cols = ['time', 'icao24', 'callsign', 'origen', 'destino', 'latitude', 'longitude', 'baro_altitude', 'vel_mps', 'vrate_mps', 'track', 'fuel_flow_kgs', 'fuel_burnt_kg']
            header = not os.path.exists(ARCHIVO_SALIDA)
            df_final[cols].to_csv(ARCHIVO_SALIDA, mode='a', header=header, index=False)
            vuelos_procesados += len(lote_actual)
            print(">>> Guardado completado con éxito. <<<")
            
        print(f"\n--- SCRIPT FINALIZADO. Total de vuelos listos en CSV: {vuelos_procesados} ---")

if __name__ == "__main__":
    generar_dataset_oficial()