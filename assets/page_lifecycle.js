(function () {
  "use strict";

  var lastPauseAt = 0;

  function setDashProps(id, props) {
    if (
      window.dash_clientside &&
      typeof window.dash_clientside.set_props === "function"
    ) {
      window.dash_clientside.set_props(id, props);
      return true;
    }
    return false;
  }

  function publishLifecycle(active, reason) {
    var payload = {
      active: active,
      reason: reason,
      ts: Date.now(),
    };
    if (!setDashProps("page-lifecycle-store", { data: payload })) {
      window.setTimeout(function () {
        setDashProps("page-lifecycle-store", { data: payload });
      }, 250);
    }
  }

  function parseStoreValue(rawValue) {
    if (!rawValue) {
      return {};
    }
    try {
      var parsed = JSON.parse(rawValue);
      if (parsed && typeof parsed === "object" && parsed.data) {
        return parsed.data;
      }
      return parsed || {};
    } catch (_err) {
      return {};
    }
  }

  function sessionIdFromLocalStorage() {
    try {
      var direct = parseStoreValue(window.localStorage.getItem("user-data-store"));
      if (direct.uuid) {
        return direct.uuid;
      }
      for (var i = 0; i < window.localStorage.length; i += 1) {
        var key = window.localStorage.key(i);
        if (!key || key.indexOf("user-data-store") === -1) {
          continue;
        }
        var value = parseStoreValue(window.localStorage.getItem(key));
        if (value.uuid) {
          return value.uuid;
        }
      }
    } catch (_err) {
      return "";
    }
    return "";
  }

  function pauseSession() {
    var now = Date.now();
    if (now - lastPauseAt < 500) {
      return;
    }
    lastPauseAt = now;

    var uuid = sessionIdFromLocalStorage();
    if (!uuid) {
      return;
    }
    var body = JSON.stringify({ uuid: uuid });
    if (navigator.sendBeacon) {
      navigator.sendBeacon(
        "/api/session/pause",
        new Blob([body], { type: "application/json" })
      );
      return;
    }
    window
      .fetch("/api/session/pause", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: body,
        keepalive: true,
        credentials: "same-origin",
      })
      .catch(function () {});
  }

  function pausePage(reason) {
    setDashProps("interval-component", { disabled: true });
    publishLifecycle(false, reason);
    pauseSession();
  }

  function resumePage(reason) {
    publishLifecycle(document.visibilityState !== "hidden", reason);
  }

  document.addEventListener("visibilitychange", function () {
    if (document.visibilityState === "hidden") {
      pausePage("visibility-hidden");
      return;
    }
    resumePage("visibility-visible");
  });

  window.addEventListener("pagehide", function () {
    pausePage("pagehide");
  });

  window.addEventListener("pageshow", function () {
    resumePage("pageshow");
  });
})();
