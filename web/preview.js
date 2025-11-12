import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TARGET_NODES = new Set([
  "FlowMatchingProgressiveUpscaler",
  "FlowMatchingStage",
]);

function ensurePreviewWidget(node, nodeData) {
  const existing = node.widgets?.find((w) => w.name === "preview");
  if (existing) {
    node.previewWidget = existing;
    return;
  }

  const widget = node.addWidget("image", "preview", null, () => {}, {
    serialize: false,
    hideOnZoom: false,
  });
  widget.nodeData = nodeData;
  node.previewWidget = widget;
}

function updatePreviewFromBlob(node, blob) {
  if (!node.previewWidget) return;
  if (node._flowMatchingPreviewURL) {
    URL.revokeObjectURL(node._flowMatchingPreviewURL);
  }
  const url = URL.createObjectURL(blob);
  node._flowMatchingPreviewURL = url;
  node.previewWidget.value = url;
}

function updatePreviewFromImage(node, image) {
  if (!node.previewWidget || !image) return;
  const format = app.getPreviewFormatParam();
  const urlParts = [
    `./view?filename=${encodeURIComponent(image.filename)}`,
    `type=${image.type}`,
    `subfolder=${encodeURIComponent(image.subfolder)}`,
    `t=${Date.now()}${format}`,
  ];
  node.previewWidget.value = urlParts.join("&");
}

function chainCallback(proto, hook, fn) {
  const original = proto[hook];
  proto[hook] = function chainedCallback(...args) {
    if (original) {
      original.apply(this, args);
    }
    fn.apply(this, args);
  };
}

app.registerExtension({
  name: "FlowMatchingUpscalerPreview",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!TARGET_NODES.has(nodeData?.name)) {
      return;
    }

    chainCallback(nodeType.prototype, "onNodeCreated", function () {
      ensurePreviewWidget(this, nodeData);

      chainCallback(this, "onRemoved", function () {
        if (this._flowMatchingPreviewURL) {
          URL.revokeObjectURL(this._flowMatchingPreviewURL);
          this._flowMatchingPreviewURL = null;
        }
      });
    });

    chainCallback(nodeType.prototype, "onExecuted", function (message) {
      if (!message) return;
      const images = message.images || message.all_outputs?.images;
      if (images?.length) {
        updatePreviewFromImage(this, images[0]);
      } else if (message.text?.length && this.previewWidget) {
        this.previewWidget.value = message.text[0];
      }
    });
  },

  init() {
    api.addEventListener("b_preview", ({ detail }) => {
      const node = app.graph?.getNodeById(app.runningNodeId);
      if (!node || !TARGET_NODES.has(node.type)) {
        return;
      }
      updatePreviewFromBlob(node, detail);
    });
  },
});
