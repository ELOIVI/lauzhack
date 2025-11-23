"""Entrenamiento del modelo de gestos desde consola."""

from __future__ import annotations

import argparse
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEQUENCE_LENGTH = 15
ALL_GESTURES = [
    "SWIPE_LEFT",
    "SWIPE_RIGHT",
    "SWIPE_UP",
    "SWIPE_DOWN",
    "PINCH_OPEN",
    "PINCH_CLOSE",
    "FIST_CLOSE",
    "OPEN_STATIC",
]

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_MCP = [2, 5, 9, 13, 17]
PALM_LANDMARKS = [0, 1, 5, 9, 13, 17]
EPSILON = 1e-6


def extract_motion_features(sequence: np.ndarray) -> Optional[np.ndarray]:
    """Replica de las features usadas por el detector en producción."""
    if sequence.shape[0] < 2:
        return None

    features: List[float] = []
    first_wrist = sequence[0][0]

    for frame_landmarks in sequence:
        normalized = frame_landmarks - first_wrist
        features.extend(normalized.flatten())

    velocities: List[float] = []
    for i in range(1, sequence.shape[0]):
        velocity = sequence[i] - sequence[i - 1]
        velocities.extend(velocity.flatten())
    features.extend(velocities)

    accelerations: List[float] = []
    for i in range(2, sequence.shape[0]):
        accel = (sequence[i] - sequence[i - 1]) - (sequence[i - 1] - sequence[i - 2])
        accelerations.extend(accel.flatten())
    features.extend(accelerations)

    wrist_trajectory = sequence[:, 0, :]
    total_displacement = wrist_trajectory[-1] - wrist_trajectory[0]
    features.extend(total_displacement.flatten())

    x_disp = float(total_displacement[0])
    y_disp = float(total_displacement[1])
    features.extend([x_disp, y_disp, float(np.sqrt(x_disp ** 2 + y_disp ** 2))])

    mean_velocity = float(
        np.mean(
            [
                np.linalg.norm(wrist_trajectory[i] - wrist_trajectory[i - 1])
                for i in range(1, wrist_trajectory.shape[0])
            ]
        )
    )
    features.append(mean_velocity)

    last_frame = sequence[-1] - first_wrist
    wrist = last_frame[0]
    palm_points = last_frame[PALM_LANDMARKS]
    palm_center = np.mean(palm_points, axis=0)
    palm_vec = palm_center - wrist
    palm_norm = float(np.linalg.norm(palm_vec[:2]) + EPSILON)

    posture_features: List[float] = []

    for tip_idx, mcp_idx in zip(FINGER_TIPS, FINGER_MCP):
        tip_vec = last_frame[tip_idx] - wrist
        tip_dist = float(np.linalg.norm(tip_vec))
        finger_length = float(np.linalg.norm(last_frame[tip_idx] - last_frame[mcp_idx]))
        extension_ratio = tip_dist / palm_norm
        angle = float(np.arctan2(tip_vec[1], tip_vec[0]))

        posture_features.extend(
            [
                tip_dist,
                finger_length,
                extension_ratio,
                float(np.sin(angle)),
                float(np.cos(angle)),
            ]
        )

    for first_idx, second_idx in zip(FINGER_TIPS[:-1], FINGER_TIPS[1:]):
        spread = float(np.linalg.norm(last_frame[first_idx] - last_frame[second_idx]))
        posture_features.append(spread)

    palm_spread = float(np.mean(np.linalg.norm(palm_points - palm_center, axis=1)))
    palm_depth = float(np.mean(np.abs(palm_points[:, 2])))

    posture_features.extend([palm_norm, palm_spread, palm_depth])

    features.extend(posture_features)

    return np.array(features, dtype=np.float32)


def load_dataset(data_dir: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Cargar todos los gestos disponibles como features listos para entrenar."""
    X: List[np.ndarray] = []
    y: List[str] = []
    counts: Dict[str, int] = {}

    print("\n" + "=" * 70)
    print("CARGANDO DATOS")
    print("=" * 70)

    for gesture in ALL_GESTURES:
        file_path = os.path.join(data_dir, f"{gesture}.pkl")

        if not os.path.exists(file_path):
            print(f"⚠️  {gesture}: sin muestras")
            continue

        with open(file_path, "rb") as fh:
            sequences = pickle.load(fh)

        valid = 0
        for sequence in sequences:
            sequence_arr = np.asarray(sequence, dtype=np.float32)
            features = extract_motion_features(sequence_arr)
            if features is None:
                continue
            X.append(features)
            y.append(gesture)
            valid += 1

        counts[gesture] = valid
        print(f"✅ {gesture}: {valid} muestras")

    print("=" * 70)
    print(f"TOTAL: {len(X)} muestras")
    print(f"Gestos: {len(counts)}/{len(ALL_GESTURES)}")
    print("=" * 70)

    return np.array(X, dtype=np.float32), np.array(y), counts


def train_model(data_dir: str, model_path: str, scaler_path: str, test_size: float, random_state: int) -> bool:
    """Ejecutar entrenamiento completo y persistir artefactos."""
    if not os.path.isdir(data_dir):
        print(f"❌ Directorio de datos no encontrado: {data_dir}")
        return False

    print("\n🚀 Iniciando entrenamiento...\n")
    X, y, counts = load_dataset(data_dir)

    if len(X) == 0:
        print("❌ No hay muestras disponibles")
        return False

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"\nTrain: {len(X_train)}")
    print(f"Test: {len(X_test)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nEntrenando Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)

    print("\n" + "=" * 70)
    print("RESULTADOS")
    print("=" * 70)
    print(f"Train: {train_acc * 100:.2f}%")
    print(f"Test: {test_acc * 100:.2f}%")

    y_pred = model.predict(X_test_scaled)
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred, labels=ALL_GESTURES)
    print("=" * 70)
    print("Confusion matrix (rows=real, cols=pred)")
    print(cm)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    with open(model_path, "wb") as fh:
        pickle.dump(model, fh)
    with open(scaler_path, "wb") as fh:
        pickle.dump(scaler, fh)

    print("\n" + "=" * 70)
    print("COMPLETADO")
    print("=" * 70)
    print(f"Modelo: {model_path}")
    print(f"Scaler: {scaler_path}")

    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenar el modelo de gestos LauzHack")
    parser.add_argument("--data-dir", default="gesture_data", help="Directorio con los .pkl de secuencias")
    parser.add_argument("--model-path", default="models/gesture_model_v3.pkl", help="Ruta de salida para el modelo")
    parser.add_argument("--scaler-path", default="models/feature_scaler_v3.pkl", help="Ruta de salida para el scaler")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proporción para test split")
    parser.add_argument("--random-state", type=int, default=42, help="Seed aleatoria para reproducibilidad")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))

    data_dir = os.path.join(base_dir, args.data_dir)
    model_path = os.path.join(base_dir, args.model_path)
    scaler_path = os.path.join(base_dir, args.scaler_path)

    success = train_model(data_dir, model_path, scaler_path, args.test_size, args.random_state)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
