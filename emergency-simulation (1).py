

import simpy
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

# ---------------------- CONFIGURACIÓN GLOBAL ----------------------

# Semilla para reproducibilidad
random.seed(9)

# Parámetros de simulación
SIM_TIME = 24 * 60  # 24 horas en minutos

# Intervalos de llegada de pacientes (en minutos)
PATIENT_INTERVALS = {
    "weekday": 15,  # Un paciente cada 15 minutos en días normales
    "weekend": 10,  # Un paciente cada 10 minutos en fines de semana
    "holiday": 8    # Un paciente cada 8 minutos en días festivos
}

# Tiempos de atención (en minutos)
TRIAGE_TIME = 10  # Tiempo de evaluación por enfermera
DOCTOR_TIME = {1: 30, 2: 25, 3: 20, 4: 15, 5: 10}  # Tiempo según severidad
LAB_TIME = 20  # Tiempo de análisis en laboratorio
XRAY_TIME = 15  # Tiempo de rayos X

# Costos de recursos (en quetzales)
DOCTOR_SALARY_PER_HOUR = 200  # Q200 por hora
NURSE_SALARY_PER_HOUR = 100   # Q100 por hora
LAB_EQUIPMENT_COST = 500000   # Q500,000 por equipo (costo inicial)
XRAY_EQUIPMENT_COST = 800000  # Q800,000 por equipo (costo inicial)
EQUIPMENT_LIFESPAN_DAYS = 5 * 365  # 5 años en días

# Variables globales para estadísticas
all_patient_times = []
lab_usage = []
xray_usage = []
doctor_usage = []
nurse_usage = []
severity_distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
waiting_times_by_severity = {1: [], 2: [], 3: [], 4: [], 5: []}

# ---------------------- CLASES Y FUNCIONES ----------------------

class Hospital:
    """Modelo del hospital con todos sus recursos y procesos"""
    
    def __init__(self, env: simpy.Environment, num_doctors: int, num_nurses: int, 
                 num_labs: int, num_xray: int):
        """
        Inicializa el hospital con sus recursos
        
        Args:
            env: Entorno de simulación
            num_doctors: Número de doctores
            num_nurses: Número de enfermeras
            num_labs: Número de laboratorios
            num_xray: Número de unidades de rayos X
        """
        self.env = env
        # Recursos con prioridad
        self.doctors = simpy.PriorityResource(env, capacity=num_doctors)
        self.nurses = simpy.PriorityResource(env, capacity=num_nurses)
        self.labs = simpy.PriorityResource(env, capacity=num_labs)
        self.xray = simpy.PriorityResource(env, capacity=num_xray)
        
        # Contadores
        self.patients_seen = 0
        self.patients_waiting = 0
        
    def patient_generator(self, interval_type: str = "weekday"):
        """
        Genera pacientes que llegan a la sala de emergencias
        
        Args:
            interval_type: Tipo de día (weekday, weekend, holiday)
        """
        i = 0
        interval = PATIENT_INTERVALS.get(interval_type, PATIENT_INTERVALS["weekday"])
            
        while True:
            # Esperar hasta el próximo paciente
            yield self.env.timeout(random.expovariate(1.0 / interval))
            
            # Crear un nuevo paciente
            i += 1
            severity = random.randint(1, 5)  # Asignar severidad aleatoria (1 es lo más grave)
            severity_distribution[severity] += 1
            
            # Registrar llegada para estadísticas
            arrival_time = self.env.now
            
            print(f"Paciente {i} llega en tiempo {arrival_time:.2f} con severidad {severity}")
            self.patients_waiting += 1
            
            # Iniciar el proceso del paciente
            self.env.process(self.patient_process(i, severity, arrival_time))
    
    def patient_process(self, patient_id: int, severity: int, arrival_time: float):
        """
        Proceso completo de atención de un paciente
        
        Args:
            patient_id: ID del paciente
            severity: Nivel de severidad (1-5, donde 1 es lo más grave)
            arrival_time: Tiempo de llegada del paciente
        """
        # Paso 1: Triage (evaluación inicial por enfermera)
        print(f"Paciente {patient_id} espera triage en tiempo {self.env.now:.2f}")
        triage_start = self.env.now
        
        with self.nurses.request(priority=severity) as req:
            yield req
            print(f"Paciente {patient_id} inicia triage en tiempo {self.env.now:.2f}")
            yield self.env.timeout(TRIAGE_TIME)
            print(f"Paciente {patient_id} termina triage en tiempo {self.env.now:.2f}")
        
        triage_time = self.env.now - triage_start
        
        # Decisión de laboratorio y rayos X (más probable para casos más graves)
        needs_lab = random.random() < (1 - (severity - 1) * 0.15)  # 100%, 85%, 70%, 55%, 40%
        needs_xray = random.random() < (1 - (severity - 1) * 0.2)  # 100%, 80%, 60%, 40%, 20%
        
        # Paso 2: Laboratorio si es necesario
        lab_time = 0
        if needs_lab:
            lab_start = self.env.now
            print(f"Paciente {patient_id} espera laboratorio en tiempo {self.env.now:.2f}")
            
            with self.labs.request(priority=severity) as req:
                yield req
                lab_usage.append((self.env.now, "start", patient_id))
                print(f"Paciente {patient_id} inicia laboratorio en tiempo {self.env.now:.2f}")
                yield self.env.timeout(LAB_TIME)
                print(f"Paciente {patient_id} termina laboratorio en tiempo {self.env.now:.2f}")
                lab_usage.append((self.env.now, "end", patient_id))
            
            lab_time = self.env.now - lab_start
        
        # Paso 3: Rayos X si es necesario
        xray_time = 0
        if needs_xray:
            xray_start = self.env.now
            print(f"Paciente {patient_id} espera rayos X en tiempo {self.env.now:.2f}")
            
            with self.xray.request(priority=severity) as req:
                yield req
                xray_usage.append((self.env.now, "start", patient_id))
                print(f"Paciente {patient_id} inicia rayos X en tiempo {self.env.now:.2f}")
                yield self.env.timeout(XRAY_TIME)
                print(f"Paciente {patient_id} termina rayos X en tiempo {self.env.now:.2f}")
                xray_usage.append((self.env.now, "end", patient_id))
            
            xray_time = self.env.now - xray_start
        
        # Paso 4: Consulta con doctor
        doctor_start = self.env.now
        print(f"Paciente {patient_id} espera doctor en tiempo {self.env.now:.2f}")
        
        with self.doctors.request(priority=severity) as req:
            yield req
            doctor_usage.append((self.env.now, "start", patient_id))
            print(f"Paciente {patient_id} inicia consulta con doctor en tiempo {self.env.now:.2f}")
            yield self.env.timeout(DOCTOR_TIME[severity])
            print(f"Paciente {patient_id} termina consulta con doctor en tiempo {self.env.now:.2f}")
            doctor_usage.append((self.env.now, "end", patient_id))
        
        doctor_time = self.env.now - doctor_start
        
        # Registrar tiempo total del paciente en el sistema
        total_time = self.env.now - arrival_time
        waiting_time = total_time - triage_time - lab_time - xray_time - doctor_time
        
        all_patient_times.append((patient_id, severity, total_time, waiting_time))
        waiting_times_by_severity[severity].append(waiting_time)
        
        print(f"Paciente {patient_id} completó todo el proceso en {total_time:.2f} minutos")
        print(f"  - Tiempo de espera: {waiting_time:.2f} minutos")
        
        self.patients_seen += 1
        self.patients_waiting -= 1


def reset_statistics():
    """Reinicia todas las variables estadísticas globales"""
    global all_patient_times, lab_usage, xray_usage, doctor_usage, nurse_usage
    global severity_distribution, waiting_times_by_severity
    
    all_patient_times = []
    lab_usage = []
    xray_usage = []
    doctor_usage = []
    nurse_usage = []
    severity_distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    waiting_times_by_severity = {1: [], 2: [], 3: [], 4: [], 5: []}


def run_simulation(num_doctors: int, num_nurses: int, num_labs: int, 
                  num_xray: int, day_type: str = "weekday") -> Dict[str, Any]:
    """
    Ejecuta la simulación con la configuración dada y devuelve estadísticas
    
    Args:
        num_doctors: Número de doctores
        num_nurses: Número de enfermeras
        num_labs: Número de laboratorios
        num_xray: Número de unidades de rayos X
        day_type: Tipo de día (weekday, weekend, holiday)
        
    Returns:
        Dict con los resultados y estadísticas de la simulación
    """
    # Reiniciar estadísticas
    reset_statistics()
    
    # Configurar simulación
    env = simpy.Environment()
    hospital = Hospital(env, num_doctors, num_nurses, num_labs, num_xray)
    
    # Iniciar generador de pacientes
    env.process(hospital.patient_generator(day_type))
    
    # Ejecutar simulación
    env.run(until=SIM_TIME)
    
    # Calcular estadísticas
    avg_time = np.mean([t[2] for t in all_patient_times]) if all_patient_times else 0
    avg_wait = np.mean([t[3] for t in all_patient_times]) if all_patient_times else 0
    patients_seen = len(all_patient_times)
    
    # Calcular tiempos promedio por severidad
    avg_time_by_severity = {}
    avg_wait_by_severity = {}
    for sev in range(1, 6):
        times = [t[2] for t in all_patient_times if t[1] == sev]
        avg_time_by_severity[sev] = np.mean(times) if times else 0
        
        waits = waiting_times_by_severity[sev]
        avg_wait_by_severity[sev] = np.mean(waits) if waits else 0
    
    # Calcular costos
    total_hours = SIM_TIME / 60  # Convertir minutos a horas
    doctor_cost = num_doctors * DOCTOR_SALARY_PER_HOUR * total_hours
    nurse_cost = num_nurses * NURSE_SALARY_PER_HOUR * total_hours
    
    # Costos de equipamiento (amortización diaria)
    lab_equipment_daily = (num_labs * LAB_EQUIPMENT_COST) / EQUIPMENT_LIFESPAN_DAYS
    xray_equipment_daily = (num_xray * XRAY_EQUIPMENT_COST) / EQUIPMENT_LIFESPAN_DAYS
    
    total_daily_cost = doctor_cost + nurse_cost + lab_equipment_daily + xray_equipment_daily
    
    # Evitar división por cero al calcular eficiencia
    efficiency = patients_seen / (total_daily_cost / 1000) if total_daily_cost > 0 else 0
    
    stats = {
        "avg_time": avg_time,
        "avg_wait": avg_wait,
        "patients_seen": patients_seen,
        "avg_time_by_severity": avg_time_by_severity,
        "avg_wait_by_severity": avg_wait_by_severity,
        "severity_distribution": severity_distribution,
        "doctor_cost": doctor_cost,
        "nurse_cost": nurse_cost,
        "lab_equipment_daily": lab_equipment_daily,
        "xray_equipment_daily": xray_equipment_daily,
        "total_daily_cost": total_daily_cost,
        "efficiency": efficiency,
        "config": {
            "doctors": num_doctors,
            "nurses": num_nurses,
            "labs": num_labs,
            "xrays": num_xray,
            "day_type": day_type
        }
    }
    
    return stats


def plot_results(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Genera gráficas con los resultados de las simulaciones
    
    Args:
        results: Lista de resultados de simulaciones
        
    Returns:
        DataFrame con el resumen de resultados
    """
    # Verificar que hay resultados
    if not results:
        print("No hay resultados para generar gráficas")
        return pd.DataFrame()
        
    # 1. Configurar subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # 2. Tiempos promedio por configuración
    configs = [f"D:{r['config']['doctors']},N:{r['config']['nurses']},L:{r['config']['labs']},X:{r['config']['xrays']}" 
               for r in results]
    avg_times = [r['avg_time'] for r in results]
    avg_waits = [r['avg_wait'] for r in results]
    
    axs[0, 0].bar(configs, avg_times, label='Tiempo total', alpha=0.7)
    axs[0, 0].bar(configs, avg_waits, label='Tiempo de espera', alpha=0.5)
    axs[0, 0].set_ylabel('Minutos')
    axs[0, 0].set_title('Tiempo promedio de pacientes por configuración')
    axs[0, 0].tick_params(axis='x', rotation=45)
    axs[0, 0].legend()
    
    # 3. Tiempos por severidad para la mejor configuración
    best_index = np.argmin([r['avg_wait'] for r in results]) if results else 0
    if best_index < len(results):
        best_config = results[best_index]
        severities = list(range(1, 6))
        severity_times = [best_config['avg_time_by_severity'][s] for s in severities]
        severity_waits = [best_config['avg_wait_by_severity'][s] for s in severities]
        
        axs[0, 1].bar(severities, severity_times, label='Tiempo total', alpha=0.7)
        axs[0, 1].bar(severities, severity_waits, label='Tiempo de espera', alpha=0.5)
        axs[0, 1].set_xlabel('Severidad (1=más grave)')
        axs[0, 1].set_ylabel('Minutos')
        axs[0, 1].set_title(f'Tiempos por severidad - Mejor config: {configs[best_index]}')
        axs[0, 1].legend()
    
    # 4. Costos por configuración
    doctor_costs = [r['doctor_cost'] for r in results]
    nurse_costs = [r['nurse_cost'] for r in results]
    equipment_costs = [r['lab_equipment_daily'] + r['xray_equipment_daily'] for r in results]
    
    axs[1, 0].bar(configs, doctor_costs, label='Doctores', alpha=0.7)
    axs[1, 0].bar(configs, nurse_costs, bottom=doctor_costs, label='Enfermeras', alpha=0.7)
    axs[1, 0].bar(configs, equipment_costs, bottom=[d+n for d,n in zip(doctor_costs, nurse_costs)], 
                  label='Equipamiento', alpha=0.7)
    axs[1, 0].set_ylabel('Costo (Q)')
    axs[1, 0].set_title('Costos diarios por configuración')
    axs[1, 0].tick_params(axis='x', rotation=45)
    axs[1, 0].legend()
    
    # 5. Eficiencia (pacientes por Q1000)
    efficiency = [r.get('efficiency', 0) for r in results]
    
    axs[1, 1].bar(configs, efficiency)
    axs[1, 1].set_ylabel('Pacientes atendidos / Q1000')
    axs[1, 1].set_title('Eficiencia por configuración')
    axs[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    try:
        plt.savefig('hospital_simulation_results.png')
        print("Gráfico guardado como 'hospital_simulation_results.png'")
    except Exception as e:
        print(f"Error al guardar gráfico: {e}")
    finally:
        plt.close()
    
    # Crear una tabla resumen
    data = []
    for i, r in enumerate(results):
        data.append([
            f"D:{r['config']['doctors']},N:{r['config']['nurses']},L:{r['config']['labs']},X:{r['config']['xrays']}",
            r['config']['day_type'],
            round(r['avg_time'], 2),
            round(r['avg_wait'], 2),
            r['patients_seen'],
            round(r['total_daily_cost'], 2),
            round(r.get('efficiency', 0), 2)
        ])
    
    df = pd.DataFrame(data, columns=['Configuración', 'Tipo de día', 'Tiempo promedio (min)', 
                                     'Tiempo espera (min)', 'Pacientes atendidos', 
                                     'Costo total (Q)', 'Eficiencia (Pacientes/Q1000)'])
    
    # Guardar tabla en CSV
    try:
        df.to_csv('hospital_simulation_results.csv', index=False)
        print("Datos guardados en 'hospital_simulation_results.csv'")
    except Exception as e:
        print(f"Error al guardar CSV: {e}")
    
    return df


def run_experiments() -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """
    Ejecuta múltiples configuraciones de simulación para encontrar la óptima
    
    Returns:
        Tupla con (resultados, dataframe_resumen)
    """
    results = []
    
    # Configuraciones a probar
    configs = [
        # (doctores, enfermeras, laboratorios, rayos X)
        (2, 3, 1, 1),  # Configuración mínima
        (3, 5, 2, 1),  # Configuración media
        (4, 6, 2, 1),  # Más personal médico
        (5, 8, 3, 2),  # Configuración alta
        (6, 10, 3, 2), # Configuración máxima
    ]
    
    print("\n--- INICIANDO SIMULACIONES DE DÍA NORMAL ---")
    # Ejecutar para día normal (weekday)
    for config in configs:
        doctors, nurses, labs, xrays = config
        print(f"\nSimulando configuración: D={doctors}, N={nurses}, L={labs}, X={xrays}")
        result = run_simulation(doctors, nurses, labs, xrays, "weekday")
        results.append(result)
    
    # Encontrar mejor configuración para día normal
    if results:
        best_config_index = np.argmin([r['avg_wait'] for r in results])
        best_config = configs[best_config_index]
        print(f"\nMejor configuración para día normal: D={best_config[0]}, N={best_config[1]}, L={best_config[2]}, X={best_config[3]}")
    else:
        best_config = (3, 5, 2, 1)  # Configuración predeterminada si no hay resultados
    
    print("\n--- SIMULACIONES PARA FINES DE SEMANA Y DÍAS FESTIVOS ---")
    # Ejecutar para fin de semana y día festivo con la mejor configuración de día normal
    for day_type in ["weekend", "holiday"]:
        doctors, nurses, labs, xrays = best_config
        print(f"\nSimulando {day_type} con config: D={doctors}, N={nurses}, L={labs}, X={xrays}")
        result = run_simulation(doctors, nurses, labs, xrays, day_type)
        results.append(result)
    
    print("\n--- SIMULACIONES CON RECURSOS AUMENTADOS PARA ALTA DEMANDA ---")
    # Probar configuraciones ajustadas para días de mayor demanda
    for day_type in ["weekend", "holiday"]:
        doctors, nurses, labs, xrays = best_config
        # Aumentar recursos para días de mayor demanda
        print(f"\nSimulando {day_type} con recursos aumentados: D={doctors+1}, N={nurses+2}, L={labs+1}, X={xrays}")
        result = run_simulation(doctors + 1, nurses + 2, labs + 1, xrays, day_type)
        results.append(result)
    
    # Generar gráficas y tabla resumen
    print("\n--- GENERANDO GRÁFICAS Y RESUMEN DE RESULTADOS ---")
    df = plot_results(results)
    
    return results, df


def print_best_configurations(results: List[Dict[str, Any]]):
    """
    Imprime un resumen de las mejores configuraciones por tipo de día
    
    Args:
        results: Lista de resultados de simulaciones
    """
    if not results:
        print("No hay resultados para analizar")
        return
        
    # Filtrar resultados por tipo de día
    weekday_results = [r for r in results if r['config']['day_type'] == 'weekday']
    weekend_results = [r for r in results if r['config']['day_type'] == 'weekend']
    holiday_results = [r for r in results if r['config']['day_type'] == 'holiday']
    
    # Encontrar la mejor configuración para cada tipo de día
    if weekday_results:
        best_weekday = min(weekday_results, key=lambda x: x['avg_wait'])
        print("\n🏥 MEJOR CONFIGURACIÓN PARA DÍA NORMAL:")
        print(f"Doctores: {best_weekday['config']['doctors']}")
        print(f"Enfermeras: {best_weekday['config']['nurses']}")
        print(f"Laboratorios: {best_weekday['config']['labs']}")
        print(f"Rayos X: {best_weekday['config']['xrays']}")
        print(f"Tiempo promedio de espera: {best_weekday['avg_wait']:.2f} minutos")
        print(f"Costo total: Q{best_weekday['total_daily_cost']:.2f}")
    
    if weekend_results:
        best_weekend = min(weekend_results, key=lambda x: x['avg_wait'])
        print("\n🏥 MEJOR CONFIGURACIÓN PARA FIN DE SEMANA:")
        print(f"Doctores: {best_weekend['config']['doctors']}")
        print(f"Enfermeras: {best_weekend['config']['nurses']}")
        print(f"Laboratorios: {best_weekend['config']['labs']}")
        print(f"Rayos X: {best_weekend['config']['xrays']}")
        print(f"Tiempo promedio de espera: {best_weekend['avg_wait']:.2f} minutos")
        print(f"Costo total: Q{best_weekend['total_daily_cost']:.2f}")
    
    if holiday_results:
        best_holiday = min(holiday_results, key=lambda x: x['avg_wait'])
        print("\n🏥 MEJOR CONFIGURACIÓN PARA DÍA FESTIVO:")
        print(f"Doctores: {best_holiday['config']['doctors']}")
        print(f"Enfermeras: {best_holiday['config']['nurses']}")
        print(f"Laboratorios: {best_holiday['config']['labs']}")
        print(f"Rayos X: {best_holiday['config']['xrays']}")
        print(f"Tiempo promedio de espera: {best_holiday['avg_wait']:.2f} minutos")
        print(f"Costo total: Q{best_holiday['total_daily_cost']:.2f}")


# ---------------------- PROGRAMA PRINCIPAL ----------------------

def main():
    """Función principal del programa"""
    print("=" * 80)
    print("SIMULADOR DE SALA DE EMERGENCIAS HOSPITALARIAS".center(80))
    print("=" * 80)
    
    try:
        results, df = run_experiments()
        
        # Mostrar resumen de resultados
        print("\n" + "=" * 80)
        print("RESUMEN DE RESULTADOS:".center(80))
        print("=" * 80)
        print(df)
        
        # Mostrar mejores configuraciones
        print("\n" + "=" * 80)
        print("MEJORES CONFIGURACIONES POR TIPO DE DÍA:".center(80))
        print("=" * 80)
        print_best_configurations(results)
        
        print("\n" + "=" * 80)
        print("SIMULACIÓN COMPLETADA CON ÉXITO".center(80))
        print("=" * 80)
        
    except Exception as e:
        print("\n" + "!" * 80)
        print(f"ERROR EN LA SIMULACIÓN: {e}")
        print("!" * 80)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()