"""Minimal Zabbix JSON-RPC client shared by the dashboard tools.

No third-party Zabbix lib — just `requests`.  The class is intentionally tiny;
it covers only what `prepare_labeling_dashboard` and `export_dashboard_items`
need.
"""
from __future__ import annotations
import logging

import requests

logger = logging.getLogger(__name__)

_API_PATH = "api_jsonrpc.php"


def _normalize_url(url: str) -> str:
    """Ensure the URL points at the JSON-RPC endpoint.

    Common Zabbix gotcha: users paste the web-UI URL (e.g. http://host/zabbix)
    instead of the API endpoint (http://host/zabbix/api_jsonrpc.php).  Hitting
    the UI returns HTML, not JSON, and json() fails cryptically.  Append the
    API path when missing.
    """
    stripped = url.rstrip("/")
    if stripped.endswith(_API_PATH):
        return stripped
    return f"{stripped}/{_API_PATH}"


class ZabbixAPI:
    def __init__(self, url: str, user: str, password: str, timeout: int = 60):
        self._url = _normalize_url(url)
        self._timeout = timeout
        self._session = requests.Session()
        self._session.proxies = {}      # bypass system proxy
        self._id = 0
        # apiinfo.version is unauthenticated, so call it before login.
        # Used to pick the right login key ('username' on 5.4+, else 'user').
        self._version = str(self._call("apiinfo.version", {}))
        token = self._login(user, password)
        # 5.4+ accepts the Authorization header; 7.2 will require it
        # (the deprecated 'auth' field in the JSON-RPC payload is being removed).
        try:
            major, minor = (int(x) for x in self._version.split(".")[:2])
        except (ValueError, IndexError):
            major, minor = 7, 0
        if (major, minor) >= (5, 4):
            self._session.headers["Authorization"] = f"Bearer {token}"
        else:
            self._auth_token_legacy = token   # injected into payloads for 5.0/5.2

    def _call(self, method: str, params: dict | list) -> object:
        self._id += 1
        payload: dict = {"jsonrpc": "2.0", "method": method, "params": params, "id": self._id}
        legacy = getattr(self, "_auth_token_legacy", None)
        if legacy:
            payload["auth"] = legacy
        resp = self._session.post(self._url, json=payload, timeout=self._timeout)
        resp.raise_for_status()
        try:
            data = resp.json()
        except ValueError as e:
            ctype = resp.headers.get("Content-Type", "?")
            snippet = (resp.text or "")[:200].replace("\n", " ").replace("\r", "")
            raise RuntimeError(
                f"Zabbix API at {self._url} returned non-JSON "
                f"(status={resp.status_code}, content-type={ctype}). "
                f"Body starts: {snippet!r}. "
                "Check that api_url points at api_jsonrpc.php and that the "
                "host/path is reachable."
            ) from e
        if "error" in data:
            raise RuntimeError(f"Zabbix API [{method}] error: {data['error']}")
        return data["result"]

    def _login(self, user: str, password: str) -> str:
        try:
            major, minor = (int(x) for x in self._version.split(".")[:2])
        except (ValueError, IndexError):
            major, minor = 7, 0
        key = "username" if (major, minor) >= (5, 2) else "user"
        return str(self._call("user.login", {key: user, "password": password}))

    def api_version(self) -> str:
        return self._version

    # ---- dashboard ----------------------------------------------------------
    def get_dashboard(self, name: str) -> dict | None:
        results = self._call(
            "dashboard.get",
            {"filter": {"name": name}, "selectPages": "extend", "selectWidgets": "extend"},
        )
        return results[0] if results else None  # type: ignore[index]

    def delete_dashboard(self, dashboardid: str) -> None:
        self._call("dashboard.delete", [dashboardid])

    def create_dashboard(self, name: str, pages: list[dict]) -> None:
        self._call("dashboard.create", {"name": name, "pages": pages})

    def update_dashboard(self, dashboardid: str, pages: list[dict]) -> None:
        self._call("dashboard.update", {"dashboardid": dashboardid, "pages": pages})

    # ---- graphs -------------------------------------------------------------
    def get_graph_items(self, graphids: list[int]) -> list[int]:
        if not graphids:
            return []
        results = self._call(
            "graphitem.get",
            {"graphids": [int(g) for g in graphids], "output": ["itemid"]},
        )
        return [int(r["itemid"]) for r in results]  # type: ignore[index]
