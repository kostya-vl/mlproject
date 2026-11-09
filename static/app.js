"use strict";

const form = document.getElementById("form");
const tokEl = document.getElementById("tok");
const spansEl = document.getElementById("spans");
const statusEl = document.getElementById("status");
const outEl = document.getElementById("out");
const submitBtn = document.getElementById("submit");

function setStatus(kind, text) {
  statusEl.hidden = false;
  statusEl.dataset.kind = kind;   // "ok" | "wait" | "error"
  statusEl.textContent = text;    // только текст, без HTML
}
function clearOutput() { outEl.textContent = ""; }

form.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  clearOutput();

  const token = tokEl.value;      // только в памяти JS, не localStorage
  const spans = spansEl.value.split("\n").map((s) => s.trim()).filter(Boolean);

  if (spans.length === 0) {
    setStatus("error", "Ошибка валидации: введите хотя бы один спан.");
    return;
  }
  if (spans.length > 1000) {
    setStatus("error", "Ошибка валидации: не больше 1000 спанов за запрос.");
    return;
  }

  submitBtn.disabled = true;
  setStatus("wait", "Отправка запроса…");

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "X-API-Key": token,
      },
      body: JSON.stringify({ spans }),
    });

    // Разбор тела — даже у ошибок сервер возвращает {"detail": ...}
    let body = null;
    try { body = await res.json(); } catch (_) { body = null; }

    if (res.status === 401) {
      setStatus("error", "Нет доступа: неверный или отсутствующий API-токен.");
      return;
    }
    if (res.status === 429) {
      const ra = res.headers.get("Retry-After");
      setStatus("error",
        `Превышен лимит запросов. Повторите через ${ra ? ra + " с" : "некоторое время"}.`);
      return;
    }
    if (res.status === 413) {
      setStatus("error", "Тело запроса слишком большое (лимит 1 МБ).");
      return;
    }
    if (res.status === 415) {
      setStatus("error", "Неподдерживаемый формат запроса.");
      return;
    }
    if (res.status === 400) {
      setStatus("error", "Некорректный JSON в запросе.");
      return;
    }
    if (res.status === 422) {
      setStatus("error", "Ошибка валидации входа (схема/типы/лимиты спанов).");
      return;
    }
    if (res.status === 503) {
      setStatus("error", "Сервис не готов обрабатывать запросы.");
      return;
    }
    if (res.status === 504) {
      setStatus("error", "Таймаут предсказания. Попробуйте меньше спанов.");
      return;
    }
    if (!res.ok) {
      // Не раскрываем internal detail, если он есть
      setStatus("error", `Ошибка сервера (HTTP ${res.status}).`);
      return;
    }

    setStatus("ok", "Готово.");
    // Недоверенный JSON — как текст, не в innerHTML
    outEl.textContent = JSON.stringify(body, null, 2);
  } catch (err) {
    setStatus("error", "Сетевая ошибка: не удалось связаться с сервисом.");
  } finally {
    submitBtn.disabled = false;
  }
});
