# -*- coding: utf-8 -*-
"""
Расчёт путевой скорости и курса с учётом ветра (треугольник скоростей).

Логика и конвенции взяты из пайплайна DeepTP (yulinliu101/DeepTP):
  - u, v — зональная (на восток) и меридиональная (на север) компоненты
    ветра в м/с, как в сетках NAM/GFS/ERA5;
  - азимуты считаются от севера по часовой стрелке, в градусах;
  - линия пути между точками — геодезическая на эллипсоиде WGS84
    (pyproj.Geod, как в tools.py DeepTP);
  - проекция ветра на курс — как в weatherMethods.getWind:
        w_along = u * sin(az) + v * cos(az),
    расширенная до полного треугольника скоростей: боковая компонента
    тоже уменьшает путевую скорость, т.к. борт держит угол сноса.

Основные функции:
  wind_triangle(v_air, u, v, track_deg)       — ядро расчёта;
  segment_correction(lat1, lon1, lat2, lon2,
                     v_air, u, v)             — то же для участка маршрута.

Все функции векторизованы: скаляры и numpy-массивы одинаковой длины
обрабатываются одинаково.
"""

import numpy as np

try:
    from pyproj import Geod

    _GEOD = Geod(ellps="WGS84")
except ImportError:  # pragma: no cover
    _GEOD = None


def wind_triangle(v_air, u, v, track_deg):
    """
    Треугольник скоростей: путевая скорость и курс для удержания линии пути.

    Параметры
    ---------
    v_air : float | ndarray
        Воздушная (истинная) скорость борта, м/с.
    u, v : float | ndarray
        Компоненты ветра, м/с: u — на восток, v — на север (конвенция NAM).
    track_deg : float | ndarray
        Заданный путевой угол (азимут линии пути), градусы от севера по
        часовой стрелке.

    Возвращает dict с полями (каждое — скаляр или ndarray):
    ---------
    w_along : попутная составляющая ветра, м/с (>0 — попутный);
    w_cross : боковая составляющая, м/с (>0 — сносит вправо от линии пути);
    v_ground : путевая скорость, м/с (NaN, если участок непроходим);
    heading_deg : курс борта (куда направить нос), градусы;
    drift_deg : угол сноса/упреждения, градусы (>0 — нос левее линии пути);
    k : коэффициент изменения скорости, v_ground / v_air;
    feasible : bool — хватает ли воздушной скорости, чтобы парировать снос.
    """
    v_air = np.asarray(v_air, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    track = np.deg2rad(np.asarray(track_deg, dtype=np.float64))

    sin_t, cos_t = np.sin(track), np.cos(track)

    # проекции ветра на линию пути (формула DeepTP + боковая компонента)
    w_along = u * sin_t + v * cos_t
    w_cross = u * cos_t - v * sin_t

    feasible = np.abs(w_cross) < v_air

    with np.errstate(invalid="ignore"):
        # угол упреждения: нос доворачивается против боковой составляющей
        wca = np.arcsin(np.where(feasible, w_cross / v_air, np.nan))
        v_ground = np.sqrt(v_air**2 - w_cross**2) + w_along

    # путевая скорость <= 0 означает, что встречный ветер сильнее борта
    feasible = feasible & (v_ground > 0)
    v_ground = np.where(feasible, v_ground, np.nan)

    heading = (np.rad2deg(track) - np.rad2deg(wca)) % 360.0

    return {
        "w_along": w_along,
        "w_cross": w_cross,
        "v_ground": v_ground,
        "heading_deg": heading,
        "drift_deg": np.rad2deg(wca),
        "k": v_ground / v_air,
        "feasible": feasible,
    }


def track_azimuth(lat1, lon1, lat2, lon2):
    """
    Путевой угол (и дистанция, м) между точками на WGS84 — как g.inv в DeepTP.
    """
    if _GEOD is None:
        raise ImportError("pyproj не установлен: pip install pyproj")
    az_fwd, _, dist = _GEOD.inv(lon1, lat1, lon2, lat2)
    return np.asarray(az_fwd) % 360.0, np.asarray(dist)


def segment_correction(lat1, lon1, lat2, lon2, v_air, u, v):
    """
    Расчёт для участка маршрута между двумя точками полётного задания.

    Кроме полей wind_triangle возвращает:
      track_deg — путевой угол участка;
      dist_m — длина участка, м;
      time_s — ожидаемое время прохождения с учётом ветра, с;
      time_still_air_s — время в штиль, с (для сравнения).
    """
    track_deg, dist_m = track_azimuth(lat1, lon1, lat2, lon2)
    res = wind_triangle(v_air, u, v, track_deg)
    res["track_deg"] = track_deg
    res["dist_m"] = dist_m
    with np.errstate(invalid="ignore"):
        res["time_s"] = dist_m / res["v_ground"]
    res["time_still_air_s"] = dist_m / np.asarray(v_air, dtype=np.float64)
    return res


if __name__ == "__main__":
    # Демонстрация: БПЛА с воздушной скоростью 20 м/с, участок на северо-восток
    demo = segment_correction(
        lat1=55.75, lon1=37.62,   # старт
        lat2=56.00, lon2=38.10,   # конец участка
        v_air=20.0,
        u=-6.0,                   # ветер: 6 м/с с востока
        v=-8.0,                   # и 8 м/с с севера (т.е. северо-восточный)
    )
    print("Путевой угол:      %6.1f°" % demo["track_deg"])
    print("Ветер вдоль пути:  %6.1f м/с" % demo["w_along"])
    print("Ветер поперёк:     %6.1f м/с" % demo["w_cross"])
    print("Путевая скорость:  %6.1f м/с (k = %.2f)" % (demo["v_ground"], demo["k"]))
    print("Курс с упреждением:%6.1f° (снос %.1f°)" % (demo["heading_deg"], demo["drift_deg"]))
    print("Время участка:     %6.0f с (в штиль %.0f с)" % (demo["time_s"], demo["time_still_air_s"]))
