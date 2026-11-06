"""
Политика аудита зависимостей (строгая):
любая находка pip-audit блокирует сборку.

Читает JSON-отчёт pip-audit, печатает таблицу «пакет / версия / ID / fix»
и завершается с кодом 1, если есть хотя бы одна уязвимость.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--report", default="metrics/pip_audit_report.json")
    args = p.parse_args()

    path = Path(args.report)
    if not path.exists():
        print(f"ERROR: отчёт {path} не найден. Сначала запустите pip-audit.", file=sys.stderr)
        return 2

    data = json.loads(path.read_text(encoding="utf-8"))
    deps = data.get("dependencies", [])

    findings = []
    for dep in deps:
        for vuln in dep.get("vulns", []):
            findings.append({
                "name": dep.get("name", "?"),
                "version": dep.get("version", "?"),
                "id": vuln.get("id", "?"),
                "fix_versions": ", ".join(vuln.get("fix_versions", [])) or "—",
            })

    if not findings:
        print("OK: pip-audit не нашёл известных уязвимостей.")
        return 0

    print(f"FAIL: найдено {len(findings)} уязвимостей (политика: любая находка блокирует сборку).\n")
    print(f"{'Пакет':<25} {'Версия':<12} {'ID':<25} {'Fix Versions'}")
    print("-" * 85)
    for f in findings:
        print(f"{f['name']:<25} {f['version']:<12} {f['id']:<25} {f['fix_versions']}")
    print("-" * 85)
    print("\nОбновите зависимости или добавьте обоснованное исключение в README.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
