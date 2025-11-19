# CLEANING/cleaning_comm_type.py
from __future__ import annotations
import re
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

class CleaningCommType:
    """
    Grupo 1: Tipo de Comunicación
    - Normaliza proto/service/attack_type
    - Valida puertos (0..65535)
    - Chequea coherencia service–proto–puerto
    - Regla del EDA:
        * Registros NORMAL incoherentes -> eliminar
        * Registros ATTACK o UNKNOWN incoherentes -> conservar pero marcar
    """

    name: str = "01_comm_type"
    depends_on: list[str] = []

    # ---- Configuración basada en el EDA ----
    _SERVICE_CANON = {
        "http":       {"proto": {"tcp"}, "ports": {80, 8080}},
        "https":      {"proto": {"tcp"}, "ports": {443, 8443}},
        "ssl":        {"proto": {"tcp"}, "ports": {443, 8443}},  # se agrupa con https
        "dns":        {"proto": {"tcp", "udp"}, "ports": {53}},
        "mqtt":       {"proto": {"tcp"}, "ports": {1883, 8883}},
        "ssh":        {"proto": {"tcp"}, "ports": {22}},
        "ftp":        {"proto": {"tcp"}, "ports": {21}},
        "dhcp":       {"proto": {"udp"}, "ports": {67, 68}},
        "ntp":        {"proto": {"udp"}, "ports": {123}},
        # rango IRC 6660-6669
        "irc":        {"proto": {"tcp"}, "ports_range": (6660, 6669)},
        # otros servicios posibles se marcarán como 'unknown' si no matchean
    }

    _UNKNOWN_TOKENS = {"-", "none", "null", "nan", "n/a", "unknown", ""}

    def _canon_str(self, s: Optional[str]) -> str:
        if s is None:
            return "unknown"
        ss = str(s).strip().lower()
        return "unknown" if ss in self._UNKNOWN_TOKENS else ss

    def _canon_proto(self, s: Optional[str]) -> str:
        p = self._canon_str(s)
        if p not in {"tcp", "udp"}:
            return "unknown"
        return p

    def _canon_service(self, s: Optional[str]) -> str:
        sv = self._canon_str(s)
        # normalizar alias obvios
        if sv in {"ssl", "tls"}:
            return "https"
        return sv

    def _canon_attack(self, s: Optional[str]) -> str:
        a = self._canon_str(s)
        # compactar etiquetas con guiones/espacios múltiples
        a = re.sub(r"\s+", "_", a)
        return a

    def _port_or_nan(self, x) -> float:
        """Convierte a número; fuera de rango -> NaN"""
        try:
            v = float(x)
        except Exception:
            return np.nan
        if np.isnan(v) or v < 0 or v > 65535:
            return np.nan
        return v

    def _irc_in_range(self, port: float) -> bool:
        if np.isnan(port):
            return False
        lo, hi = self._SERVICE_CANON["irc"]["ports_range"]
        return lo <= int(port) <= hi

    def _service_rule_ok(self, service: str, proto: str, port_resp: float) -> bool:
        """Valida coherencia service–proto–puerto destino (id.resp_p) según tabla EDA."""
        rule = self._SERVICE_CANON.get(service)
        if not rule:
            # si el servicio no está en la tabla: no imponemos restricción estricta
            return True
        # proto coherente
        if proto not in rule.get("proto", {"tcp", "udp"}):
            return False
        # puertos coherentes
        if "ports" in rule and len(rule["ports"]) > 0:
            if np.isnan(port_resp) or int(port_resp) not in rule["ports"]:
                return False
        if "ports_range" in rule:
            if not self._irc_in_range(port_resp):
                return False
        return True

    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        df = df.copy()

        # === 1) Normalización básica ===
        before_rows = len(df)

        for col, fn in [
            ("proto", self._canon_proto),
            ("service", self._canon_service),
            ("attack_type", self._canon_attack),
        ]:
            if col in df.columns:
                df[col] = df[col].map(fn)
            else:
                # si falta, crear como unknown (evita fallos posteriores)
                df[col] = "unknown"

        # Unificar services desconocidos
        df["service"] = df["service"].where(df["service"] != "", "unknown")
        if "ssl" in df["service"].unique():
            df["service"] = df["service"].replace({"ssl": "https"})

        # === 2) Puertos válidos ===
        for col in ("id.orig_p", "id.resp_p"):
            if col in df.columns:
                df[col] = df[col].map(self._port_or_nan)
            else:
                # si no existe, crear como NaN (no debería pasar)
                df[col] = np.nan

        # === 3) Proto desconocido: mantener pero marcar como unknown ===
        df["proto"] = df["proto"].where(df["proto"].isin({"tcp", "udp"}), "unknown")

        # === 4) Coherencia service–proto–puerto destino ===
        incoh_mask = pd.Series(False, index=df.index)
        # Solo evaluar coherencia cuando tengamos al menos 'service' y 'proto'
        if {"service", "proto", "id.resp_p"}.issubset(df.columns):
            incoh_mask = ~df.apply(
                lambda r: self._service_rule_ok(
                    r.get("service", "unknown"),
                    r.get("proto", "unknown"),
                    r.get("id.resp_p", np.nan),
                ),
                axis=1,
            )

        # === 5) Criterio diferenciado según clase (EDA):
        # - normal + incoherente -> eliminar
        # - attack/unknown + incoherente -> conservar pero marcar
        attack_col = "attack_type"
        if attack_col not in df.columns:
            # fallback por si el dataset original trae otro nombre:
            attack_col = "Attack_type" if "Attack_type" in df.columns else "attack_type"

        # normalizar posibles valores 'normal'
        df[attack_col] = df[attack_col].fillna("unknown")
        # Algunos datasets usan 'benign' o 'normal'
        df[attack_col] = df[attack_col].replace(
            {"benign": "normal", "normal_traffic": "normal"}
        )

        is_normal = df[attack_col].eq("normal")
        drop_mask = incoh_mask & is_normal

        dropped_rows = int(drop_mask.sum())
        df = df.loc[~drop_mask].copy()

        # Para el resto (ataque/unknown incoherentes) marcamos bandera
        incoh_after = incoh_mask.loc[df.index]
        df["comm_incoherent"] = incoh_after.fillna(False)

        # === 6) Proto–puerto gross mismatches adicionales (según EDA):
        # proto=udp con puertos típicos tcp-only (80,443,22,21,8080,8443)
        # proto=tcp con puertos udp-only (67,68,123) -> aplicar lógica como arriba
        tcp_only = {80, 8080, 443, 8443, 22, 21}
        udp_only = {67, 68, 123}

        gross_incoh = (
            (df["proto"].eq("udp") & df["id.resp_p"].isin(tcp_only)) |
            (df["proto"].eq("tcp") & df["id.resp_p"].isin(udp_only))
        )

        # misma política que antes:
        drop_mask2 = gross_incoh & df[attack_col].eq("normal")
        dropped_rows += int(drop_mask2.sum())
        df = df.loc[~drop_mask2].copy()
        # los que quedan (attack/unknown) se marcan
        df.loc[gross_incoh.index.intersection(df.index), "comm_incoherent"] = (
            df.loc[gross_incoh.index.intersection(df.index), "comm_incoherent"] | gross_incoh.loc[gross_incoh.index.intersection(df.index)]
        )

        # === 7) Completar NaNs residuales de proto/service con 'unknown' ===
        for c in ["proto", "service"]:
            df[c] = df[c].fillna("unknown")

        # === 8) Reporte ===
        report: Dict[str, Any] = {
            "rows_before": before_rows,
            "rows_after": len(df),
            "dropped": before_rows - len(df),
            "dropped_by_incoherence": dropped_rows,
            "unknown_proto_count": int((df["proto"] == "unknown").sum()),
            "unknown_service_count": int((df["service"] == "unknown").sum()),
            "comm_incoherent_kept": int(df["comm_incoherent"].sum()),
            "notes": [
                "proto/service/attack_type normalizados a minúsculas",
                "puertos validados [0..65535]",
                "coherencia service–proto–puerto: normal incoherente eliminado; attack/unknown conservado y marcado",
                "bandera 'comm_incoherent' añadida",
            ],
        }
        return df, report
