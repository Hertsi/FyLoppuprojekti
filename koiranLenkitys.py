import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.fft import fft
from math import radians, cos, sin, sqrt, atan2
import folium
from streamlit_folium import st_folium

url1 = "https://raw.githubusercontent.com/Hertsi/FyLoppuprojekti/refs/heads/main/LinearAcceleration.csv"
url2 = "https://raw.githubusercontent.com/Hertsi/FyLoppuprojekti/refs/heads/main/Location.csv"
accel_data = pd.read_csv(url1)
gps_data = pd.read_csv(url2)

st.title('koiran iltalenkki')

#matalapäästösuodatus
def butter_lowpass_filter(data, cutoff, nyq, order=4):
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

fs = 60 
cutoff = 3  #katkaisutaajuus Hz (tiukempi suodatus kohinan vähentämiseksi)
nyq = 0.5 * fs

#x-komponentti
accel_x = accel_data['Linear Acceleration x (m/s^2)']
filtered_signal_x = butter_lowpass_filter(accel_x, cutoff, nyq)

#zoomtyökalu
start_time = st.sidebar.slider('alkuaika (s)', 0, int(accel_data['Time (s)'].max()), 0)
end_time = st.sidebar.slider('loppuaika (s)', 0, int(accel_data['Time (s)'].max()), int(accel_data['Time (s)'].max()))

mask = (accel_data['Time (s)'] >= start_time) & (accel_data['Time (s)'] <= end_time)
zoomed_time = accel_data['Time (s)'][mask]
zoomed_accel_x = accel_x[mask]
zoomed_filtered_signal_x = filtered_signal_x[mask]

#suodatettu signaali
st.subheader("suodatettu kiihtyvyys")
fig, ax = plt.subplots()
ax.plot(zoomed_time, zoomed_accel_x, label='raakadata x komponentti')
ax.plot(zoomed_time, zoomed_filtered_signal_x, label='suodatettu x komponentti', color='red')
ax.set_xlabel('aika (s)')
ax.set_ylabel('kiihtyvyys (m/s^2)')
ax.set_title('suodatettu kiihtyvyys ja askelten määrän laskenta (<-- zoom vasemmalla)')
ax.grid(True)
ax.legend()
st.pyplot(fig)

#askelmäärä suodatetusta kiihtyvyysdatasta
step_count_filtered = np.floor(np.sum((filtered_signal_x[:-1] < 0) & (filtered_signal_x[1:] > 0)) / 2)
st.write(f"**askelmäärä laskettuna suodatetusta kiihtyvyysdatasta:** {step_count_filtered}")

#askelmäärä fourieranalyysin perusteella
#käytetään hanningikkunaa vähentämään kohinaa
window = np.hanning(len(accel_x))
accel_x_windowed = accel_x * window

#fourieranalyysi ja tehon spektri raakadatan perusteella
N = len(accel_x_windowed)
T = 1.0 / fs
yf = fft(accel_x_windowed)
xf = np.fft.fftfreq(N, T)[:N//2]
psd = 2.0/N * np.abs(yf[0:N//2])

valid_range = (xf >= 1) & (xf <= 10)
xf = xf[valid_range]
psd = psd[valid_range]

st.subheader("tehospektri (x komponentti, 1 - 10 Hz)")
fig, ax = plt.subplots()
ax.plot(xf, psd)
ax.set_title('tehospektri (x komponentti raakadata)')
ax.set_xlabel('taajuus (Hz)')
ax.set_ylabel('teho')
ax.grid()
st.pyplot(fig)

#askeltiheys
peak_freq = xf[np.argmax(psd)]
step_time = 1 / peak_freq if peak_freq != 0 else np.inf

#askeleiden määrä fourieranalyysin perusteella
total_time = accel_data['Time (s)'].iloc[-1] - accel_data['Time (s)'].iloc[0]
step_count_fourier = total_time / step_time
st.write(f"**askelmäärä laskettuna Fourier-analyysin perusteella:** {step_count_fourier:.0f}")

#matkan laskeminen GPS datasta (haversinen kaava)
def haversine(lon1, lat1, lon2, lat2):
    R = 6371000  # Maan säde metreinä
    phi1, phi2 = radians(lat1), radians(lat2)
    delta_phi = radians(lat2 - lat1)
    delta_lambda = radians(lon2 - lon1)
    a = sin(delta_phi/2)**2 + cos(phi1) * cos(phi2) * sin(delta_lambda/2)**2
    return R * (2 * atan2(sqrt(a), sqrt(1 - a)))

#kokonaismatkan laskeminen
total_distance = 0
coords = []
for i in range(1, len(gps_data)):
    total_distance += haversine(gps_data['Longitude (°)'][i-1], gps_data['Latitude (°)'][i-1],
                                gps_data['Longitude (°)'][i], gps_data['Latitude (°)'][i])
    coords.append((gps_data['Latitude (°)'][i], gps_data['Longitude (°)'][i]))

#keskinopeus
time_seconds = gps_data['Time (s)'].iloc[-1] - gps_data['Time (s)'].iloc[0]
speed_m_s = total_distance / time_seconds
speed_kmh = speed_m_s * 3.6

st.write(f"**kuljettu matka:** {total_distance:.2f} metriä")
st.write(f"**keskinopeus:** {speed_kmh:.2f} km/h")

#askelpituus suodatetusta kiihtyvyysdatasta
step_length_filtered = total_distance / step_count_filtered if step_count_filtered != 0 else 0
st.write(f"**askelpituus suodatetusta kiihtyvyysdatasta:** {step_length_filtered:.2f} metriä")

#askelpituus fourieranalyysin perusteella
step_length_fourier = total_distance / step_count_fourier if step_count_fourier != 0 else 0
st.write(f"**askelpituus Fourier-analyysin perusteella:** {step_length_fourier:.2f} metriä")

#kartta
st.subheader("reitti kartalla")
start_coords = [gps_data['Latitude (°)'].iloc[0], gps_data['Longitude (°)'].iloc[0]]
map_ = folium.Map(location=start_coords, zoom_start=15)
folium.PolyLine(coords, color="blue", weight=2.5, opacity=1).add_to(map_)

#kartta streamlitissä
st_folium(map_, width=700, height=500)