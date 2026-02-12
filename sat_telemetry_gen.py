"""
Synthetic satellite telemetry generator with annotated injecting of anomalies.

Generates signals for various sensors with different types of anomalies.
Can inlude both sensor-specific and environmental-based anomalies.

AI programming tools for code completion and code refactoring have been used in this script.

Author: Vojtech Orava
Date: 2026
"""

import argparse
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import ast

@dataclass
class SensorSpec:
    name: str
    base_func: Callable[[np.ndarray], np.ndarray]  
    noise_std: float
    bounds: Optional[Tuple[float, float]] = None  # optional physical bounds (min, max)


# anomaly signal disortions
def inject_spike(signal: np.ndarray, idx: int, duration: int, magnitude: float = 5.0):
    """
    Injects a random magnitude spike into the sensor signal at a given index.

    Parameters
    ----------
    signal : np.ndarray
        The sensor signal to be modified
    idx : int
        The index at which to start the spike
    duration : int
        The number of samples for which to maintain the spike
    magnitude : float, optional
        The magnitude of the spike to inject (default is 5.0)

    Returns
    -------
    np.ndarray
        The modified signal
    """
    sign = 1 if random.random() < 0.5 else -1
    for i in range(idx, min(idx + duration, len(signal))):
        signal[i] += sign * magnitude
    return signal


def inject_stuck_at(signal: np.ndarray, idx: int, duration: int, value: float = 0.0):
    """
    Sets the sensor signal values to a given value at a specific index
    for a specified duration.

    Parameters
    ----------
    signal : np.ndarray
        The sensor signal to be modified
    idx : int
        The index at which to start the stuck value
    duration : int
        The number of samples for which to maintain the stuck value
    value : float, optional
        The value to which to set the signal (default is 0.0)

    Returns
    -------
    np.ndarray
        The modified signal
    """
    end = min(len(signal), idx + duration)
    signal[idx:end] = value
    return signal


def inject_drift(signal: np.ndarray, idx: int, duration: int, drift_per_step: float = 0.01):
    """
    Injects a drift into the sensor signal at a given index.

    Parameters
    ----------
    signal : np.ndarray
        The sensor signal to be modified
    idx : int
        The index at which to start the drift
    duration : int
        The number of samples for which to maintain the drift
    drift_per_step : float, optional
        The amount of drift to apply per step (default is 0.01)

    Returns
    -------
    np.ndarray
        The modified signal
    """
    end = min(len(signal), idx + duration)
    steps = np.arange(0, end - idx)
    signal[idx:end] = signal[idx:end] + steps * drift_per_step
    return signal


def inject_increased_noise(signal: np.ndarray, idx: int, duration: int, extra_std: float = 1.0):
    """
    Temporarily injects increased noise into the sensor signal at a given index.

    Parameters
    ----------
    signal : np.ndarray
        The sensor signal to be modified
    idx : int
        The index at which to start the increased noise
    duration : int
        The number of samples for which to maintain the increased noise
    extra_std : float, optional
        The amount of extra standard deviation to add to the noise (default is 1.0)

    Returns
    -------
    np.ndarray
        The modified signal
    """
    end = min(len(signal), idx + duration)
    signal[idx:end] = signal[idx:end] + np.random.normal(scale=extra_std, size=(end - idx,))
    return signal


def inject_frozen(signal: np.ndarray, idx: int, duration: int):
    """
    Temporarily freezes the signal values at a given index.

    Parameters
    ----------
    signal : np.ndarray
        The sensor signal to be modified
    idx : int
        The index at which to start the frozen values
    duration : int
        The number of samples for which to maintain the frozen values

    Returns
    -------
    np.ndarray
        The modified signal
    """
    end = min(len(signal), idx + duration)
    signal[idx:end] = signal[idx]
    return signal

def inject_decreased(signal: np.ndarray, idx: int, duration: int, factor: float = 0.8):
    """
    Temporarily decrease the signal values by a multiplicative factor.

    Parameters
    ----------
    signal : np.ndarray
        Original sensor signal array.
    idx : int
        Start index of the anomaly.
    duration : int
        Number of samples to modify.
    factor : float, optional
        Multiplicative decrease factor (default 0.8 = 20% drop).
    """
    end = min(len(signal), idx + duration)
    signal[idx:end] = signal[idx:end] * factor
    return signal

# dictionary of anomaly functions
ANOMALY_FUNCS = {
    "spike": inject_spike,
    "stuck_at": inject_stuck_at,
    "drift": inject_drift,
    "increased_noise": inject_increased_noise,
    "frozen": inject_frozen,
    "decreased": inject_decreased
}

class TelemetryGenerator:
    def __init__(self, start_time_hours: float = 0.0, days: float = 1.0, hz: float = 1.0, seed: Optional[int] = None,
                 illumination_cycles_per_day: float = 2.0):
        """
        start_time_hours: float starting time in hours (arbitrary); used to compute sinusoidal baselines
        days: how many days of data to generate
        hz: sampling frequency (samples per second)
        illumination_cycles_per_day: how many illumination cycles (high/low) occur per 24h
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.start_h = start_time_hours
        self.days = days
        self.hz = hz
        self.s_per_sample = 1.0 / hz
        self.total_seconds = int(days * 24 * 3600)
        self.N = int(self.total_seconds * hz)
        self.t_seconds = np.arange(0, self.N) * self.s_per_sample
        # convert to hours for daily-period baselines
        self.t_hours = self.start_h + (self.t_seconds / 3600.0)
        self.illum_period_hours = 24.0 / illumination_cycles_per_day
        self.sensors: Dict[str, SensorSpec] = {}

    def register_sensor(self, spec: SensorSpec):
        self.sensors[spec.name] = spec

    def generate_baseline(self) -> pd.DataFrame:
        """
        Generates baseline (normal) telemetry data for all registered sensors.

        The generated data will have columns for timestamp in seconds, time in hours,
        and one column per registered sensor. The values in the sensor columns
        are the baseline signal plus Gaussian noise.

        Returns
        -------
        pd.DataFrame
            Baseline telemetry data
        """
        data = {"timestamp_s": self.t_seconds, "time_h": self.t_hours}
        # timestamp_s is seconds since the start of the generated data
        # time_h is hours since the start of the generated data = (timestamp_s/3600)
        for name, spec in self.sensors.items():
            base = spec.base_func(self.t_hours)
            noise = np.random.normal(scale=spec.noise_std, size=base.shape)
            sig = base + noise
            if spec.bounds is not None:
                sig = np.clip(sig, spec.bounds[0], spec.bounds[1])
            data[name] = sig
        return pd.DataFrame(data)

    def inject_anomalies(self, df: pd.DataFrame, anomaly_plan: List[Dict]) -> pd.DataFrame:
        """
        Injects anomalies into the telemetry data according to the anomaly plan.

        The anomaly plan is a list of dictionaries, where each dictionary
        contains the following keys:
        - "sensor": the name of the sensor to inject the anomaly into
        - "type": the type of anomaly to inject ("spike", "stuck_at", "drift", "increased_noise", "frozen", "decreased")
        - "start_idx": the starting index of the anomaly (in samples)
        - "duration": the duration of the anomaly (in samples)
        - "kwargs": additional keyword arguments to pass to the anomaly function
        
        Example: {"sensor":..., "type":..., "start_idx":..., "duration":..., kwargs...}

        The function injects the anomalies into a copy of the input DataFrame and
        returns the modified DataFrame.

        The function also labels the injected anomalies in the returned DataFrame
        by setting the "anomaly" column to "anomalous" and the "anomaly_type" column
        to the type of anomaly injected, and the "anomaly_sensor" column to the name
        of the sensor into which the anomaly was injected.

        If a sensor is not present in the input DataFrame, the function prints a warning
        and skips the anomaly.

        If the type of anomaly is not recognized, the function prints a warning and
        skips the anomaly.

        Parameters
        ----------
        df : pd.DataFrame
            Telemetry data to inject anomalies into
        anomaly_plan : List[Dict]
            List of dictionaries containing anomaly parameters

        Returns
        -------
        pd.DataFrame
            Modified telemetry data with injected anomalies
        """
        df = df.copy()
        df["anomaly"] = "normal"
        df["anomaly_type"] = "none"
        df["anomaly_sensor"] = "none"
        for a in anomaly_plan:
            sensor = a["sensor"]
            
            if sensor not in df.columns:
                print(f"Warning: sensor {sensor} not present; skipping anomaly")
                continue
            typ = a.get("type", "spike")
            func = ANOMALY_FUNCS.get(typ)
            
            if func is None:
                print(f"Unknown anomaly type {typ}; skipping")
                continue
            
            start_idx = a.get("start_idx", 0)
            duration = a.get("duration", 1)
            col = df[sensor].to_numpy().copy()
    
            kwargs = a.get("kwargs", {})
            
            try:
                new_col = func(col, start_idx, duration, **kwargs)
            
            except Exception:
                # try generic calling (some funcs have different signatures)
                new_col = func(col, start_idx, duration)
            
            if "pct" in sensor: # percentage has to be positive
                df[sensor] = np.abs(new_col)
            else:
                df[sensor] = new_col
            # label the duration
            end = min(len(df), start_idx + duration)
            df.loc[start_idx:end - 1, "anomaly"] = "anomalous"
            df.loc[start_idx:end - 1, "anomaly_type"] = typ
            # handle anomaly_sensor logic
            df.loc[start_idx:end - 1, "anomaly_sensor"] = df.loc[start_idx:end - 1, "anomaly_sensor"].apply(
                lambda x: sensor if x == "none" else f"{x},{sensor}" if sensor not in x.split(',') else x
            )
            
        return df



def build_default_sensors(gen: TelemetryGenerator) -> List[SensorSpec]:
    """
    Builds a list of default SensorSpec objects for use with the TelemetryGenerator.

    These SensorSpec objects are pre-defined to represent common types of
    sensors found on space probes. They are designed to be used
    as-is or with minimal modification to generate realistic-looking
    telemetry data.

    The sensors include:

    - Solar panel temperatures (high when illuminated, low when in shadow)
    - Battery voltage (charges while illuminated, discharges in dark)
    - CPU temperature (depends on activity and solar heating)
    - Magnetometer (3 axes) - near-Earth varying geomagnetic field + noise
    - Sun sensor: high when illuminated, near zero in shadow
    - Gyro rates - small angular rate noise around zero
    - Communication link signal strength - depends on attitude and illumination
    - Radiation counts (higher in certain orbits or SAA, model as Poisson around low mean)
    - Communication bus noise in percentage
    - Telescope sensor voltage
    - Atomic clock frequency drift over time in microseconds
    - Memory usage in percentage
    - Reaction wheel speed (rpm) over time t_h [hours]

    The behavior of each sensor is documented in its corresponding of the SensorSpec object.
    """
    def illumination(t_h: np.ndarray):
        # smooth approx of day/night with a sinusoid of specified period
        return 0.5 * (1 + np.sin(2 * np.pi * t_h / gen.illum_period_hours))

    sensors: List[SensorSpec] = []

    # Solar panel temperature: high when illuminated, low when in shadow
    def solar_temp(t_h):
        illum = illumination(t_h)
        # mid temp 20C, vary +-15C with illumination
        return 20.0 + 15.0 * illum + 2.0 * np.sin(2 * np.pi * t_h / 24.0)  # small daily drift

    sensors.append(SensorSpec("solar_panel_temp_C_1", base_func=solar_temp, noise_std=0.5, bounds=( -40, 125)))
    sensors.append(SensorSpec("solar_panel_temp_C_2", base_func=solar_temp, noise_std=0.5, bounds=( -40, 125)))
    sensors.append(SensorSpec("solar_panel_temp_C_3", base_func=solar_temp, noise_std=0.5, bounds=( -40, 125)))
    sensors.append(SensorSpec("solar_panel_temp_C_4", base_func=solar_temp, noise_std=0.5, bounds=( -40, 125)))

    # Battery voltage: charges while illuminated, discharges in dark; slow dynamics
    def battery_voltage(t_h):
        illum = illumination(t_h)
        return 3.7 + 0.3 * illum - 0.05 * np.sin(2 * np.pi * t_h / 12.0)

    sensors.append(SensorSpec("battery_voltage_V", base_func=battery_voltage, noise_std=0.01, bounds=(3.0, 4.2)))

    # CPU temperature: depends on activity and solar heating
    def cpu_temp(t_h):
        return 35.0 + 5.0 * np.sin(2 * np.pi * t_h / 24.0) + 3.0 * np.random.randn(*t_h.shape) * 0.0

    sensors.append(SensorSpec("cpu_temp_C", base_func=cpu_temp, noise_std=0.3, bounds=(-20, 85)))

    # Magnetometer (3 axes) - near-Earth varying geomagnetic field + noise
    def mag_x(t_h):
        noise = np.random.normal(scale=500, size=t_h.shape)
        return (20000.0 * np.cos(2 * np.pi * t_h / 1.0)  + noise )* 1e-9  

    def mag_y(t_h):
        noise = np.random.normal(scale=500, size=t_h.shape)
        return (20000.0 * np.sin(2 * np.pi * t_h / 1.0) + noise )* 1e-9

    def mag_z(t_h):
        noise = np.random.normal(scale=500, size=t_h.shape)
        return (10000.0 * np.sin(2 * np.pi * t_h / 0.5) + noise )* 1e-9

    sensors.append(SensorSpec("mag_x_T", base_func=mag_x, noise_std=5e-9, bounds=(-1e-6, 1e-6)))
    sensors.append(SensorSpec("mag_y_T", base_func=mag_y, noise_std=5e-9, bounds=(-1e-6, 1e-6)))
    sensors.append(SensorSpec("mag_z_T", base_func=mag_z, noise_std=5e-9, bounds=(-1e-6, 1e-6)))

    # Sun sensor: high when illuminated, near zero in shadow
    def sun_sensor(t_h):
        illum = illumination(t_h)
        return 1000.0 * illum  # arbitrary units

    sensors.append(SensorSpec("sun_sensor_counts", base_func=sun_sensor, noise_std=10.0, bounds=(0.0, 2000.0)))

    # Gyro rates - small angular rate noise around zero
    def gyro_x(t_h):
        return 0.01 * np.sin(2 * np.pi * t_h / 0.1)

    sensors.append(SensorSpec("gyro_x_deg_s", base_func=gyro_x, noise_std=0.005, bounds=(-5.0, 5.0)))

    # Communication link signal strength - depends on attitude and illumination (not strictly physical but plausible)
    def comms_snr(t_h):
        # produce occasional deep fades (multiplicative) as baseline low-frequency variations
        return 20.0 + 5.0 * np.sin(2 * np.pi * t_h / 6.0) + 10.0 * illumination(t_h)

    sensors.append(SensorSpec("comms_snr_dB", base_func=comms_snr, noise_std=0.5, bounds=(-50.0, 60.0)))

    # Radiation counts (higher in certain orbits or SAA, model as Poisson around low mean)
    def rad_counts(t_h):
        # occasional low-frequency bumps to simulate SAA
        mean_counts = 0.5 + 2.0 * (0.5 * (1 + np.sin(2 * np.pi * t_h / 24.0)))
        return np.random.poisson(mean_counts)

    sensors.append(SensorSpec("rad_counts_cps_1", base_func=rad_counts, noise_std=0.2, bounds=(0.0, 1000.0)))
    sensors.append(SensorSpec("rad_counts_cps_2", base_func=rad_counts, noise_std=0.2, bounds=(0.0, 1000.0)))
    
    # communication bus packet loss in percentage
    def bus_packet_loss(t_h):
        return np.abs(1.0 * np.sin(2 * np.pi * t_h /1.0) * 1e-3)

    sensors.append(SensorSpec("bus_packet_loss_pct_1", base_func=bus_packet_loss, noise_std=5e-3, bounds=(0.0, 1.0)))
    sensors.append(SensorSpec("bus_packet_loss_pct_2", base_func=bus_packet_loss, noise_std=5e-3, bounds=(0.0, 1.0)))
    
    # Telescope sensor voltage with base level at 5 V, two observation windows per 24 hours
    def telescope_sensor_voltage(t_h):
        # observation schedule: 2 windows per day, each 2h
        phase = (t_h % 24)  # time of day in hours
        observing = ((phase >= 4) & (phase < 6)) | ((phase >= 16) & (phase < 18))
        obs_factor = observing.astype(float)  # 1 when active, else 0

        # baseline + daily drift + voltage sag when active + noise
        base_voltage = 5.0
        daily_drift = 0.02 * np.sin(2 * np.pi * t_h / 24.0)
        voltage_sag = -0.1 * obs_factor  # 100 mV drop during active period
        noise = np.random.normal(scale=0.005, size=t_h.shape)

        return base_voltage + daily_drift + voltage_sag + noise

    sensors.append(SensorSpec("telescope_sensor_V", base_func=telescope_sensor_voltage, noise_std=0.002, bounds=(4.5, 5.2)))

    # Simulate atomic clock frequency drift over time in microseconds.
    def atomic_clock_drift(t_h):
        bias = np.random.normal(0.0, 0.005)  # constant offset, ±0.005 ppb
        aging_rate = 0.0001  # ppb per day (drift)
        daily_temp_amp = 0.002  # daily temperature effect, ±0.002 ppb
        noise_std = 0.0005  # short-term jitter
        
        aging = aging_rate * (t_h / 24.0)  # drift accumulates slowly
        temp_effect = daily_temp_amp * np.sin(2 * np.pi * t_h / 24.0)
        noise = np.random.normal(0.0, noise_std, size=t_h.shape)        
        freq_drift = bias + aging + temp_effect + noise

        # occasional resync corrections (every ~48h) 
        resync_interval = 48.0  # hours
        correction_strength = 0.01  # ppb
        for i in range(len(t_h)):
            if (t_h[i] % resync_interval) < 0.1:
                freq_drift[i:] -= correction_strength  # small step correction
                
        dt_s = np.mean(np.diff(t_h)) * 3600  # convert hours to seconds
        freq_error_frac = freq_drift * 1e-9  # ppb → fractional frequency offset
        time_error_s = np.cumsum(freq_error_frac * dt_s)  # integrate drift
        time_error_us = time_error_s * 1e6

        return time_error_us  

    sensors.append(SensorSpec("atomic_clock_drift_us", base_func=atomic_clock_drift, noise_std=0.0001, bounds=(-0.05, 0.05)))
    
    def memory_usage_pct(t_h):
        base = 30.0 + 10.0 * np.sin(2 * np.pi * t_h / 24.0) # small diurnal load
        # add slow upward trend over mission time
        trend = 0.01 * (t_h / 24.0) # 0.01% per day
        mu = base + trend
        # occasionally simulate GC/reset events: subtract 15-30% every few days
        mu = mu.astype(float)
        return mu


    sensors.append(SensorSpec("memory_usage_pct", base_func=memory_usage_pct, noise_std=0.5, bounds=(0.0, 100.0)))

    # Simulate reaction wheel speed (rpm), slow oscillations, random corrections, occasional attitude maneuvers
    def reaction_wheel_speed(t_h):
        base = 1500 + 500 * np.sin(2 * np.pi * t_h / 1.5)  # rpm
        # add random micro-corrections (noise)
        noise = np.random.normal(0, 30, size=len(t_h))
        # add occasional ramps to simulate attitude slews
        wheel = base + noise
        for i in range(3, len(t_h), 500):
            if np.random.rand() < 0.05:  # 5% chance of maneuver start
                ramp_duration = np.random.randint(50, 200)
                ramp_magnitude = np.random.choice([-800, 800])  # spin up or down
                ramp = np.linspace(0, ramp_magnitude, ramp_duration)
                end_i = min(i + ramp_duration, len(t_h))
                wheel[i:end_i] += ramp[: end_i - i]
                
                recover_i = min(end_i + 100, len(t_h))
                recovery = np.linspace(ramp_magnitude, 0, recover_i - end_i)
                wheel[end_i:recover_i] += recovery

        return wheel
    
    sensors.append(SensorSpec("reaction_wheel_speed", base_func=reaction_wheel_speed, noise_std=3.0, bounds=(-2000.0, 2000.0)))

    return sensors


def generate_environmental_anomalies(total_len: int):
    """
    Generate a list of environmental anomaly events affecting multiple sensors simultaneously.

    Returns:
        list of anomaly plans
    """
    anomalies = []

    def rand_window(min_dur=500, max_dur=4000):
        start = random.randint(0, total_len - max_dur - 1)
        duration = random.randint(min_dur, max_dur)
        return start, duration

    # 1. Solar Storm Event 
    # Radiation spike, magnetometer noise, SNR drop, CPU temp rise
    start, dur = rand_window(2000, 5000)
    anomalies += [
        {"sensor": "rad_counts_cps_1", "type": "spike", "start_idx": start, "duration": dur,
         "kwargs": {"magnitude": 20.0}},
        {"sensor": "rad_counts_cps_2", "type": "spike", "start_idx": start, "duration": dur,
         "kwargs": {"magnitude": 25.0}},
        {"sensor": "mag_x_T", "type": "increased_noise", "start_idx": start, "duration": dur,
         "kwargs": {"extra_std": 1e-7}},
        {"sensor": "mag_y_T", "type": "increased_noise", "start_idx": start, "duration": dur,
         "kwargs": {"extra_std": 1e-7}},
        {"sensor": "mag_x_T", "type": "increased_noise", "start_idx": start, "duration": dur,
         "kwargs": {"extra_std": 1e-7}},
        {"sensor": "mag_z_T", "type": "increased_noise", "start_idx": start, "duration": dur,
         "kwargs": {"extra_std": 1e-7}},
        {"sensor": "comms_snr_dB", "type": "decreased", "start_idx": start, "duration": dur,
         "kwargs": {"factor": 0.5}},
        {"sensor": "cpu_temp_C", "type": "increased_noise", "start_idx": start, "duration": dur,
         "kwargs": {"extra_std": 2.0}},
    ]

    # 2. Earth Eclipse 
    # No sunlight - solar panels, sun sensor, battery voltage drop
    start, dur = rand_window(1500, 3500)
    anomalies += [
        {"sensor": s, "type": "decreased", "start_idx": start, "duration": dur,
         "kwargs": {"factor": 0.85}}
        for s in ["sun_sensor_counts", "solar_panel_temp_C_1", "solar_panel_temp_C_2",
                  "solar_panel_temp_C_3", "solar_panel_temp_C_4"]
    ]
    anomalies.append({"sensor": "battery_voltage_V", "type": "drift", "start_idx": start, "duration": dur,
                      "kwargs": {"drift_per_step": -0.005}})

    # 3. Reaction Wheel Imbalance 
    # Mechanical oscillation - higher noise & transient drift in attitude sensors
    start, dur = rand_window(1000, 2500)
    anomalies += [
        {"sensor": "reaction_wheel_speed", "type": "increased_noise", "start_idx": start,
         "duration": dur, "kwargs": {"extra_std": 200}},
        {"sensor": "gyro_x_deg_s", "type": "increased_noise", "start_idx": start,
         "duration": dur, "kwargs": {"extra_std": 0.1}},
    ]

    # 4. Software Fault / Memory Leak 
    # Gradual increase in CPU temp, memory, bus noise, slower reaction wheel
    start, dur = rand_window(3000, 6000)
    anomalies += [
        {"sensor": "memory_usage_pct", "type": "drift", "start_idx": start, "duration": dur,
         "kwargs": {"drift_per_step": 0.02}},
        {"sensor": "cpu_temp_C", "type": "drift", "start_idx": start, "duration": dur,
         "kwargs": {"drift_per_step": 0.01}},
        {"sensor": "bus_packet_loss_pct_1", "type": "increased_noise", "start_idx": start, "duration": dur,
         "kwargs": {"extra_std": 0.02}},
        {"sensor": "bus_packet_loss_pct_2", "type": "increased_noise", "start_idx": start, "duration": dur,
         "kwargs": {"extra_std": 0.02}},
    ]

    # 5. Telescope Overload (observation during solar flare) 
    # Telescope voltage dips, CPU and memory spike, radiation rises
    start, dur = rand_window(800, 2000)
    anomalies += [
        {"sensor": "telescope_sensor_V", "type": "decreased", "start_idx": start,
         "duration": dur, "kwargs": {"factor": 0.9}},
        {"sensor": "rad_counts_cps_1", "type": "spike", "start_idx": start, "duration": dur,
         "kwargs": {"magnitude": 10.0}},
        {"sensor": "cpu_temp_C", "type": "spike", "start_idx": start, "duration": dur,
         "kwargs": {"magnitude": 5.0}},
        {"sensor": "memory_usage_pct", "type": "spike", "start_idx": start, "duration": dur,
         "kwargs": {"magnitude": 15.0}},
    ]

    return anomalies


def auto_plan_anomalies(gen: TelemetryGenerator, sensors: List[SensorSpec], 
                        n_anomalies: int = 10, min_duration_s: int = 60,
                        max_duration_s: int = 3600) -> List[Dict]:
    """
    Automatically generate a list of anomaly plans for a given set of sensors.

    Each anomaly plan consists of a dictionary containing the following keys:
    - sensor: str, name of the sensor to be affected
    - type: str, type of anomaly (spike, stuck_at, drift, increased_noise, decreased, frozen)
    - start_idx: int, index at which the anomaly starts
    - duration: int, duration of the anomaly in samples
    - kwargs: dict, additional arguments specific to the chosen anomaly type

    Parameters:
    - gen: TelemetryGenerator, the telemetry generator to use
    - sensors: List[SensorSpec], list of sensors to be considered
    - n_anomalies: int, number of anomaly plans to generate (default: 10)
    - min_duration_s: int, minimum duration of anomalies in seconds (default: 60)
    - max_duration_s: int, maximum duration of anomalies in seconds (default: 3600)

    Returns:
    - List[Dict], list of anomaly plans
    """
    plan = []
    N = gen.N
    for i in range(n_anomalies):
        sensor = random.choice(sensors)
        typ = random.choice(list(ANOMALY_FUNCS.keys()))
        start_idx = random.randint(0, max(0, N - 1))
        dur_s = random.randint(min_duration_s, max_duration_s)
        duration = int(dur_s * gen.hz)
        
        
        kwargs = {}
        if typ == "spike":
            kwargs = {"magnitude": random.uniform(1.0 * sensor.noise_std, 10.0 * sensor.noise_std)}
        elif typ == "stuck_at":
            kwargs = {"value": random.uniform(sensor.bounds[0], sensor.bounds[1])}
        elif typ == "drift":
            kwargs = {"drift_per_step": random.uniform(0.0001, 0.1)}
        elif typ == "increased_noise":
            kwargs = {"extra_std": random.uniform(0.5, 5.0)}
        elif typ == "decreased":
            kwargs = {"factor": random.uniform(0.1, 0.9)}
        elif typ == "frozen":
            kwargs = {}
        plan.append({"sensor": sensor.name, "type": typ, "start_idx": start_idx, "duration": duration, "kwargs": kwargs})
    return plan


def load_plan(path: str) -> List[Dict]:
    def parse_dict_string(x):
        """
        Safely parses a string containing a Python dictionary literal into a dictionary.
        Needed for kwargs column in CSV

        If the string is empty or malformed, returns an empty dictionary.

        Parameters:
        - x: str, string containing a Python dictionary literal

        Returns:
        - dict, the parsed dictionary
        """
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            # return an empty dict if the cell is empty or malformed
            return {}

    df = pd.read_csv(path, converters={"kwargs": parse_dict_string})
    
    return df.to_dict("records")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=float, default=1.0, help="the number of days to simulate")
    p.add_argument("--hz", type=float, default=1.0, help="sampling frequency (Hz) of generated data (1Hz = 1 sample per second)")
    p.add_argument("--seed", type=int, default=0, help="seed for randomness")
    p.add_argument("--out", type=str, default="synthetic_sat_telemetry.csv", help="output CSV file path")
    p.add_argument("--cycles_per_day", type=float, default=2.0, help="illumination cycles per day (default 2)")
    p.add_argument("--auto_anomalies", type=int, default=8, help="if > 0, auto-insert N of anomalies randomly")
    p.add_argument("--env_anomalies", type=int, default=1, help="if > 0, inserts a pack of environment anomalies N times")
    p.add_argument("--use_plan", type=str, default="", help="path to anomaly plan CSV file, prioritized over auto_anomalies and env_anomalies")
    args = p.parse_args()

    gen = TelemetryGenerator(days=args.days, hz=args.hz, seed=args.seed, illumination_cycles_per_day=args.cycles_per_day)
    sensors = build_default_sensors(gen)
    for s in sensors:
        gen.register_sensor(s)

    # generates default sensors behavior
    df = gen.generate_baseline()

    # anomaly plan is prioritized over anomaly generation    
    if args.use_plan != "":
        try:
            plan = load_plan(args.use_plan)
            print(f"Successfully loaded anomaly plan from file {args.use_plan}.")
        except FileNotFoundError:
            print(f"Could not find anomaly plan file {args.use_plan}.")
            exit(1)
        
    else:
        plan = []
        print(f"No anomaly plan file provided, generating {args.auto_anomalies} random anomalies and {args.env_anomalies} environment anomalies.")
        if args.auto_anomalies > 0:
            plan = auto_plan_anomalies(gen, sensors, n_anomalies=args.auto_anomalies,
                                    min_duration_s=10, max_duration_s=3600)
            
        for _ in range(args.env_anomalies):
            env_anomalies = generate_environmental_anomalies(len(df))
            plan = plan + env_anomalies

    df = gen.inject_anomalies(df, plan)

    # save plan and df as csv file
    plan_df = pd.DataFrame(plan)
    plan_csv = args.out.replace('.csv', '_anomaly_plan.csv')
    plan_df.to_csv(plan_csv, index=False)
    df.to_csv(args.out, index=False)

    print(f"Saved telemetry to {args.out}.")
    print(f"Saved anomaly plan to {plan_csv}.")

if __name__ == '__main__':
    main()
