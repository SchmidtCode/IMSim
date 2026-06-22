(function () {
  "use strict";

  function setDashProps(id, props) {
    if (
      window.dash_clientside &&
      typeof window.dash_clientside.set_props === "function"
    ) {
      try {
        window.dash_clientside.set_props(id, props);
        return true;
      } catch (_err) {
        return false;
      }
    }
    return false;
  }

  document.addEventListener(
    "toggle",
    function (event) {
      var target = event.target;
      if (!target || target.id !== "lesson-snapshot-disclosure") {
        return;
      }
      setDashProps("lesson-snapshot-open-store", {
        data: {
          open: Boolean(target.open),
          ts: Date.now(),
        },
      });
    },
    true
  );
})();
