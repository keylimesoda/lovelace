import QtQuick
import QtQuick.Controls
import QtQuick.Effects
import Quickshell
import Quickshell.Io
import qs.Ui
import qs.Commons
import "Model.js" as Model

Panel {
  id: root
  moduleName: "keylimesoda.lovelace"
  ipcTarget: "keylimesoda.lovelace"
  // manageIpc: false so this panel can own the single IpcHandler the target
  // permits — needed for the extra `service` and `vision` methods below.
  manageIpc: false

  // Helper that drives and reads the local inference server. It ships in the
  // plugin directory (see `lovelace-monitor`); engine selection, container
  // names and the API endpoint live at the top of that script.
  readonly property string bin: {
    const u = Qt.resolvedUrl("lovelace-monitor").toString()
    return u.startsWith("file://") ? u.slice(7) : u
  }
  readonly property color fg: root.bar ? root.bar.foreground : Color.foreground
  readonly property string ff: root.bar ? root.bar.fontFamily : Style.font.family
  readonly property color urgentColor: root.bar ? root.bar.urgent : Color.urgent

  // ---- service state, polled from the helper ----
  property string serviceState: "checking"   // off | starting | ready | failed
  property string stateBeforeAction: ""
  property bool actionRunning: false
  property var info: ({})

  // ---- live rates from the request log's windowed throughput sample (feed) ----

  readonly property bool on: root.serviceState === "ready"
  readonly property bool starting: root.serviceState === "starting"
  readonly property bool failed: root.serviceState === "failed"
  readonly property bool serviceActive: root.on || root.starting
  readonly property bool busy: root.actionRunning
  readonly property bool visionOn: root.info.vision === "on"
  readonly property bool specOn: root.info.spec === "dflash2"
  // DFlash2/MTP switching is a ninfer launch option; lovelace profiles pin MTP.
  readonly property bool specAvailable: root.info.engine === "ninfer"

  // Model headline (from /v1/models) over a runtime tag (runtime name, spec
  // method + draft tokens, KV type); the runtime stands in before the first
  // info sample names the model.
  readonly property string runtimeName: root.info.runtime || root.info.engine || "lovelace"
  readonly property string headline: Model.modelName(root.info) || root.runtimeName
  readonly property string tag: Model.runtimeTag(root.info)

  readonly property color stateColor: root.on ? Color.accent
    : root.failed ? root.urgentColor
    : root.starting ? root.fg
    : Color.muted

  readonly property string stateLabel: {
    if (root.actionRunning)
      return (root.stateBeforeAction === "ready" || root.stateBeforeAction === "starting")
        ? "Stopping" : "Starting"
    if (root.on) return "Ready"
    if (root.failed) return "Failed"
    if (root.starting) return "Starting"
    if (root.serviceState === "checking") return "Checking"
    return "Offline"
  }

  readonly property real ctxFraction: Model.ctxDeviceFraction(root.info)

  // Keyboard cursor targets: "hero" (power switch) | "vision" (input switch) |
  // "actions" (dashboard / logs buttons, selected by actionIndex)
  property string focusSection: "hero"
  property int actionIndex: 0
  property bool cursorActive: false

  function pollStatus() { if (!statusProc.running) statusProc.running = true }
  function refreshInfo() { if (!infoProc.running) infoProc.running = true }

  function ingestInfo(raw) {
    const text = String(raw || "").trim()
    if (!text) return
    const parsed = Model.parseInfo(text)
    root.info = parsed
    root.serviceState = parsed.state
  }

  function toggleService() {
    if (root.actionRunning) return
    root.stateBeforeAction = root.serviceState
    root.actionRunning = true
    actionProc.command = [root.bin, "toggle"]
    actionProc.running = true
  }

  function openLogs() {
    if (!logsProc.running) logsProc.running = true
  }

  function openDashboard() {
    if (!dashProc.running) dashProc.running = true
  }

  function toggleVision() {
    if (root.actionRunning) return
    root.stateBeforeAction = root.serviceState
    root.actionRunning = true
    actionProc.command = [root.bin, "vision", "toggle"]
    actionProc.running = true
  }

  function toggleSpec() {
    if (root.actionRunning) return
    root.stateBeforeAction = root.serviceState
    root.actionRunning = true
    actionProc.command = [root.bin, "spec", "toggle"]
    actionProc.running = true
  }

  IpcHandler {
    target: "keylimesoda.lovelace"

    function open() { root.open() }
    function close() { root.close() }
    function show() { root.open() }
    function hide() { root.close() }
    function toggle() { root.toggle() }
    function service() { root.toggleService() }
    function vision() { root.toggleVision() }
    function spec() { root.toggleSpec() }
  }

  onOpenedChanged: {
    if (opened) {
      root.refreshInfo()
      root.cursorActive = false
      root.focusSection = "hero"
      root.actionIndex = 0
    }
  }

  Process {
    id: statusProc
    command: [root.bin, "status"]
    stdout: StdioCollector {
      waitForEnd: true
      onStreamFinished: {
        const value = String(text || "").trim()
        if (["ready", "starting", "failed", "off"].indexOf(value) >= 0)
          root.serviceState = value
      }
    }
  }

  Process {
    id: infoProc
    command: [root.bin, "info"]
    stdout: StdioCollector {
      waitForEnd: true
      onStreamFinished: root.ingestInfo(text)
    }
  }

  Process {
    id: actionProc
    onExited: {
      root.actionRunning = false
      root.pollStatus()
      root.refreshInfo()
    }
  }

  Process {
    id: logsProc
    command: ["omarchy", "launch", "terminal", root.bin, "logs"]
  }

  Process {
    id: dashProc
    command: ["omarchy", "launch", "browser", "http://127.0.0.1:8080/"]
  }

  Timer { interval: 5000; repeat: true; running: true; onTriggered: root.pollStatus() }

  Timer {
    interval: 4000
    repeat: true
    running: root.opened
    triggeredOnStart: true
    onTriggered: root.refreshInfo()
  }

  Component.onCompleted: {
    root.pollStatus()
    root.refreshInfo()
  }

  implicitWidth: button.implicitWidth
  implicitHeight: button.implicitHeight

  BarIconButton {
    id: button
    anchors.fill: parent
    bar: root.bar
    // Official Qwen mark (monochrome silhouette) tinted by state color.
    iconComponent: qMarkBar
    active: root.on || root.failed
    activeColor: root.failed ? root.urgentColor : Color.accent
    dimmed: root.serviceState === "off"
    tooltipText: root.failed ? root.runtimeName + " failed. Click for details, right-click to retry."
      : root.on ? root.headline + " is ready on " + root.runtimeName + ". Click for details, right-click to stop."
      : root.starting ? root.runtimeName + " is starting. Click for details, right-click to stop."
      : root.runtimeName + " is offline. Click for details, right-click to start."
    onPressed: function(b) {
      if (b === Qt.RightButton) root.toggleService()
      else root.toggle()
    }
  }

  KeyboardPanel {
    id: panel
    anchorItem: button
    owner: root
    bar: root.bar
    open: root.opened
    focusTarget: keyCatcher
    contentWidth: panel.fittedContentWidth(Style.space(380))
    contentHeight: panel.fittedContentHeight(panelColumn.implicitHeight, Style.space(1400))

    PanelKeyCatcher {
      id: keyCatcher
      anchors.fill: parent
      onMoveRequested: function(dx, dy) {
        if (!root.cursorActive) { root.cursorActive = true; return }
        const sections = root.specAvailable ? ["hero", "spec", "vision", "actions"] : ["hero", "vision", "actions"]
        if (root.focusSection === "actions") {
          // Inside actions: horizontal picks dashboard/logs, vertical leaves.
          if (dx !== 0) {
            root.actionIndex = dx > 0 ? 1 : 0
            return
          }
          if (dy === 0) return
          root.actionIndex = 0
          root.focusSection = dy > 0 ? "hero" : "vision"
          return
        }
        const d = dy !== 0 ? dy : dx
        if (d === 0) return
        const i = sections.indexOf(root.focusSection)
        root.focusSection = sections[(i + (d > 0 ? 1 : sections.length - 1)) % sections.length]
        root.actionIndex = 0
      }
      onActivateRequested: {
        if (!root.cursorActive) return
        if (root.focusSection === "hero") root.toggleService()
        else if (root.focusSection === "spec") root.toggleSpec()
        else if (root.focusSection === "vision") root.toggleVision()
        else if (root.actionIndex === 0) root.openDashboard()
        else root.openLogs()
      }
      onCloseRequested: root.close()
      onTabRequested: function(direction) { root.switchPanel(direction) }

      ScrollView {
        id: scrollArea
        anchors.fill: parent
        clip: true
        ScrollBar.horizontal.policy: ScrollBar.AlwaysOff
        ScrollBar.vertical.policy: panelColumn.implicitHeight > height ? ScrollBar.AsNeeded : ScrollBar.AlwaysOff

        Column {
          id: panelColumn
          width: scrollArea.availableWidth
          spacing: Style.space(14)

          // ---------- Hero: Qwen mark · model [state] / runtime tag · power switch ----------
          PanelHero {
            iconComponent: qMarkHero
            title: root.headline
            detail: root.stateLabel.toUpperCase()
            meta: root.tag
            foreground: root.fg
            fontFamily: root.ff

            trailingControl: Component {
              ToggleSwitch {
                id: heroSwitch
                checked: root.serviceActive
                busy: root.busy
                hasCursor: root.cursorActive && root.focusSection === "hero"
                foreground: root.fg
                onHovered: function(on) {
                  if (on) {
                    root.cursorActive = true
                    root.focusSection = "hero"
                  }
                }
                onToggled: root.toggleService()

                PanelToolTip {
                  visible: heroSwitch.containsMouse
                  text: (root.serviceActive ? "Stop " : "Start ") + root.runtimeName
                  fontFamily: root.ff
                }
              }
            }
          }

          // ---------- Live at-a-glance strip ----------
          Row {
            width: parent.width
            spacing: Style.space(12)

            StatCell { label: "Generation"; value: root.on ? Model.rateLabel(root.info, "decode_tps") : "—" }
            StatCell { label: "Prefill"; value: root.on ? Model.rateLabel(root.info, "prefill_tps") : "—" }
            StatCell { label: "Power"; value: root.on ? Model.watts(root.info.power_watts) : "—" }
          }

          PanelSeparator {
            foreground: root.fg
          }

          // ---------- Model ----------
          Column {
            width: parent.width
            spacing: Style.space(6)

            PanelSectionHeader {
              text: "MODEL"
              foreground: root.fg
              fontFamily: root.ff
            }

            Row {
              width: parent.width
              spacing: Style.space(20)

              Column {
                width: (parent.width - parent.spacing) / 2
                spacing: Style.spacing.labelGap

                InfoPair { label: "Model ID"; value: root.info.model_id || "—" }
                InfoPair { label: "Context"; value: Model.contextLabel(root.info) }
                InfoPair { label: "KV cache"; value: Model.kvLabel(root.info) }
                InfoPair { label: "Spec decode"; value: Model.specLabel(root.info) }
              }

              Column {
                width: (parent.width - parent.spacing) / 2
                spacing: Style.spacing.labelGap

                InfoPair { label: "GPU"; value: Model.gpuLabel(root.info) }
                InfoPair { label: "Uptime"; value: root.on ? Model.uptime(root.info.epoch) : "—" }
                InfoPair { label: "Endpoint"; value: root.info.endpoint || "—" }
                InfoPair { label: "Image"; value: root.info.image || "—" }
              }
            }
          }

          // ---------- Throughput (live) ----------
          PanelSeparator {
            visible: root.on
            foreground: root.fg
          }

          Column {
            width: parent.width
            spacing: Style.space(6)
            visible: root.on

            PanelSectionHeader {
              text: "THROUGHPUT"
              foreground: root.fg
              fontFamily: root.ff
            }

            Row {
              width: parent.width
              spacing: Style.space(20)

              Column {
                width: (parent.width - parent.spacing) / 2
                spacing: Style.spacing.labelGap

                InfoPair { label: "Prefix cache"; value: Model.prefixHitLabel(root.info) }
                InfoPair { label: "Draft accept"; value: Model.specAcceptLabel(root.info) }
                InfoPair { label: "Generated"; value: Model.generatedLabel(root.info) }
              }

              Column {
                width: (parent.width - parent.spacing) / 2
                spacing: Style.spacing.labelGap

                InfoPair { label: "Requests"; value: Model.requestsLabel(root.info) }
                InfoPair { label: "Power draw"; value: Model.watts(root.info.power_watts) }
                InfoPair { label: "Total requests"; value: Model.group(root.info.requests_total) }
              }
            }
          }

          // ---------- Continuation cache (live) ----------
          PanelSeparator {
            visible: root.on
            foreground: root.fg
          }

          Column {
            width: parent.width
            spacing: Style.space(8)
            visible: root.on

            PanelSectionHeader {
              text: "CONTEXT CACHE"
              foreground: root.fg
              fontFamily: root.ff
            }

            InfoPair { label: "Device KV"; value: Model.ctxDeviceLabel(root.info) }

            // Device KV occupancy bar (paged cache, from /telemetry).
            Item {
              width: parent.width
              implicitHeight: Style.space(10)

              Rectangle {
                anchors.fill: parent
                radius: height / 2
                color: Util.alpha(root.fg, 0.12)
              }

              Rectangle {
                anchors.verticalCenter: parent.verticalCenter
                anchors.left: parent.left
                height: parent.height
                width: Math.max(0, parent.width * root.ctxFraction)
                radius: parent.height / 2
                color: Color.accent
              }
            }

            InfoPair { label: "Host spill"; value: Model.ctxHostLabel(root.info) }
            InfoPair { label: "State"; value: Model.ctxStateLabel(root.info) }
            InfoPair { label: "Pressure"; value: Model.ctxPressureLabel(root.info) }
          }

          // ---------- Spec decode ----------
          PanelSeparator {
            visible: root.specAvailable
            foreground: root.fg
          }

          Column {
            visible: root.specAvailable
            width: parent.width
            spacing: Style.space(8)

            PanelSectionHeader {
              text: "SPEC DECODE"
              foreground: root.fg
              fontFamily: root.ff
            }

            Text {
              textFormat: Text.PlainText
              width: parent.width
              wrapMode: Text.Wrap
              color: root.fg
              opacity: 0.65
              font.family: root.ff
              font.pixelSize: Style.font.bodySmall
              text: "DFlash2 K7 speculative decoding (pinned companion weights). Off falls back to MTP3 (3-draft), which fits the full 262K context."
            }

            Text {
              visible: Model.specImpact(root.info) !== ""
              textFormat: Text.PlainText
              width: parent.width
              wrapMode: Text.Wrap
              color: root.fg
              font.family: root.ff
              font.pixelSize: Style.font.body
              text: Model.specImpact(root.info)
            }

            Toggle {
              id: specToggle
              width: parent.width
              label: "Enable DFlash2"
              checked: root.specOn
              foreground: root.fg
              fontFamily: root.ff
              hasCursor: root.cursorActive && root.focusSection === "spec"
              onHovered: function(on) {
                if (on) {
                  root.cursorActive = true
                  root.focusSection = "spec"
                }
              }
              onClicked: root.toggleSpec()
            }
          }

          // ---------- Vision input ----------
          PanelSeparator {
            foreground: root.fg
          }

          Column {
            width: parent.width
            spacing: Style.space(8)

            PanelSectionHeader {
              text: "VISION"
              foreground: root.fg
              fontFamily: root.ff
            }

            Text {
              textFormat: Text.PlainText
              width: parent.width
              wrapMode: Text.Wrap
              color: root.fg
              opacity: 0.65
              font.family: root.ff
              font.pixelSize: Style.font.bodySmall
              text: "Multimodal input: the model accepts images and video frames."
            }

            Text {
              visible: Model.visionImpact(root.info) !== ""
              textFormat: Text.PlainText
              width: parent.width
              wrapMode: Text.Wrap
              color: root.fg
              font.family: root.ff
              font.pixelSize: Style.font.body
              text: Model.visionImpact(root.info)
            }

            Toggle {
              id: visionToggle
              width: parent.width
              label: "Enable vision"
              checked: root.visionOn
              foreground: root.fg
              fontFamily: root.ff
              hasCursor: root.cursorActive && root.focusSection === "vision"
              onHovered: function(on) {
                if (on) {
                  root.cursorActive = true
                  root.focusSection = "vision"
                }
              }
              onClicked: root.toggleVision()
            }
          }

          // ---------- Actions ----------
          PanelSeparator {
            foreground: root.fg
          }

          Row {
            width: parent.width
            spacing: Style.space(8)

            Button {
              id: dashBtn
              width: (parent.width - Style.space(8)) / 2
              // Explicit height: guards against the button collapsing when its
              // implicit height resolves late (icon-glyph metrics), which would
              // drop the whole actions section out of the column's implicit
              // height and clip it off the panel.
              height: implicitHeight > 0
                ? implicitHeight
                : Style.spacing.controlHeight + Style.spacing.controlPaddingY * 2
              iconText: "\u{F056E}"  // view dashboard
              text: "Dashboard"
              foreground: root.fg
              fontFamily: root.ff
              bordered: true
              hasCursor: root.cursorActive && root.focusSection === "actions" && root.actionIndex === 0
              onClicked: root.openDashboard()
              onHovered: function(h) {
                if (h) {
                  root.cursorActive = true
                  root.focusSection = "actions"
                  root.actionIndex = 0
                }
              }
            }

            Button {
              id: logsBtn
              width: (parent.width - Style.space(8)) / 2
              height: implicitHeight > 0
                ? implicitHeight
                : Style.spacing.controlHeight + Style.spacing.controlPaddingY * 2
              iconText: "\u{F00C5}"  // console
              text: "Logs"
              foreground: root.fg
              fontFamily: root.ff
              bordered: true
              hasCursor: root.cursorActive && root.focusSection === "actions" && root.actionIndex === 1
              onClicked: root.openLogs()
              onHovered: function(h) {
                if (h) {
                  root.cursorActive = true
                  root.focusSection = "actions"
                  root.actionIndex = 1
                }
              }
            }
          }
        }
      }
    }
  }

  // ---------- Reusable inline components ----------

  // The official Qwen mark: a monochrome line drawing of the brand mark
  // (assets/qwen.png), tinted to the service state color.
  component QwenMark: Item {
    id: mark
    property real size: Style.space(20)
    property color ink: root.fg

    width: size
    height: size

    MultiEffect {
      anchors.fill: parent
      colorization: 1.0
      colorizationColor: mark.ink

      Behavior on colorizationColor { ColorAnimation { duration: 200 } }

      source: Image {
        smooth: true
        sourceSize: Qt.size(200, 200)
        source: "assets/qwen.png"
        fillMode: Image.PreserveAspectFit
      }
    }
  }

  // Bar-sized mark.
  Component {
    id: qMarkBar
    QwenMark {
      size: Style.bar.iconCanvas
      ink: root.stateColor
    }
  }

  // Hero-sized mark; pulses while the service is waking up.
  Component {
    id: qMarkHero
    QwenMark {
      size: Style.font.display
      ink: root.stateColor

      SequentialAnimation on opacity {
        running: root.starting && root.opened
        loops: Animation.Infinite
        alwaysRunToEnd: true
        NumberAnimation { from: 1.0; to: 0.55; duration: 950; easing.type: Easing.InOutSine }
        NumberAnimation { from: 0.55; to: 1.0; duration: 950; easing.type: Easing.InOutSine }
      }
    }
  }

  // Vertical stat chip: bold value over a dim label.
  component StatCell: Item {
    id: statCell
    property string label: ""
    property string value: ""

    width: parent ? (parent.width - parent.spacing * 2) / 3 : 0
    implicitHeight: statInner.implicitHeight

    Column {
      id: statInner
      spacing: Style.space(2)

      Text {
        textFormat: Text.PlainText
        text: statCell.value
        color: root.fg
        font.family: root.ff
        font.pixelSize: Style.font.body
        font.bold: true
        opacity: statCell.value === "—" ? 0.45 : 1.0
      }

      Text {
        textFormat: Text.PlainText
        text: statCell.label
        color: root.fg
        opacity: 0.55
        font.family: root.ff
        font.pixelSize: Style.font.caption
        font.letterSpacing: 0.8
      }
    }
  }

  // Label left, flexible gap, value right.
  component InfoPair: Row {
    property string label: ""
    property string value: ""

    width: parent.width
    spacing: Style.space(8)

    InfoLabel { text: label }
    Item { width: Math.max(0, parent.width - parent.children[0].implicitWidth - parent.children[2].implicitWidth - parent.spacing * 2); height: 1 }
    InfoValue { text: value }
  }

  component InfoLabel: Text {
    textFormat: Text.PlainText
    color: root.fg
    opacity: 0.6
    font.family: root.ff
    font.pixelSize: Style.font.bodySmall
  }

  component InfoValue: Text {
    textFormat: Text.PlainText
    color: root.fg
    font.family: root.ff
    font.pixelSize: Style.font.bodySmall
    elide: Text.ElideRight
  }
}