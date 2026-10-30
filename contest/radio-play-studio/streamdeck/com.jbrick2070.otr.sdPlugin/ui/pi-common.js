// SPDX-License-Identifier: MIT
/* global WebSocket, window, document */
// Shared by the four property inspectors: the Stream Deck socket, the
// Studio status banner, the port field, and the "Open the Studio" link.

(function () {
  let ws = null;
  let uuid = "";
  let actionInfo = {};
  let currentSettings = {};
  let bindControlsFn = function () {};

  function send(payload) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(payload));
    }
  }

  function saveActionSettings(partial) {
    currentSettings = Object.assign({}, currentSettings, partial);
    send({ event: "setSettings", context: uuid, payload: currentSettings });
  }

  function sendToPlugin(payload) {
    send({ action: actionInfo.action, event: "sendToPlugin", context: uuid, payload: payload });
  }

  var PHASE_WORDS = {
    off: ["err", "The Studio is not running (nothing answers on this port)."],
    loading: ["warn", "Reading the form from the Studio…"],
    form_error: ["err", "The form could not be read."],
    ready: ["ok", "Ready: the Studio answers and the form is in hand."],
    starting: ["warn", "Sending the press to the Studio…"],
    running: ["ok", "An episode is being made."],
    busy: ["warn", "The GPU is held by something the deck cannot follow."],
    done: ["ok", "The last episode is ready to play."],
    error: ["err", "The last run failed."]
  };

  function showState(state) {
    var banner = document.getElementById("studio-status");
    var detail = document.getElementById("studio-detail");
    var portEl = document.getElementById("studio-port");
    var words = PHASE_WORDS[state.phase] || ["warn", String(state.phase || "")];
    if (banner) {
      banner.className = "status-banner status-" + words[0];
      banner.textContent = words[1];
    }
    if (detail) {
      var reason = state.formError || state.busyReason || state.runError || "";
      detail.textContent = reason;
      detail.style.display = reason ? "block" : "none";
    }
    if (portEl && document.activeElement !== portEl) {
      portEl.value = String(state.port || 8765);
    }
  }

  function bindCommon() {
    var portEl = document.getElementById("studio-port");
    if (portEl) {
      portEl.addEventListener("change", function () {
        sendToPlugin({ event: "updatePort", port: Number(portEl.value) });
      });
    }
    var openEl = document.getElementById("open-studio");
    if (openEl) {
      openEl.addEventListener("click", function (e) {
        e.preventDefault();
        sendToPlugin({ event: "openStudio" });
      });
    }
  }

  window.piRegisterControls = function (fn) {
    bindControlsFn = fn;
  };

  window.connectElgatoStreamDeckSocket = function (inPort, inPropertyInspectorUUID, inRegisterEvent, inInfo, inActionInfo) {
    uuid = inPropertyInspectorUUID;
    actionInfo = JSON.parse(inActionInfo || "{}");
    currentSettings = (actionInfo.payload && actionInfo.payload.settings) || {};

    ws = new WebSocket("ws://127.0.0.1:" + inPort);
    ws.onopen = function () {
      send({ event: inRegisterEvent, uuid: uuid });
      bindCommon();
      bindControlsFn(currentSettings, saveActionSettings);
      sendToPlugin({ event: "piReady" });
    };
    ws.onmessage = function (evt) {
      var msg = {};
      try {
        msg = JSON.parse(evt.data);
      } catch (err) {
        return;
      }
      if (msg.event === "sendToPropertyInspector" && msg.payload && msg.payload.event === "studioState") {
        showState(msg.payload);
      } else if (msg.event === "didReceiveSettings" && msg.payload) {
        currentSettings = msg.payload.settings || {};
        bindControlsFn(currentSettings, saveActionSettings);
      }
    };
  };
})();
