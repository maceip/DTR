const platforms = {
  android: {
    status: 'KOTLIN SDK · DEVICE CHECKED', name: 'Native execution, on the phone.',
    description: 'Embed the Kotlin SDK. It selects compatible model artifacts for the engines in your app and runs inference locally.',
    boundary: 'One unchanged Pixel APK used downloaded adapters to run LiteRT CPU, ExecuTorch CPU, and native LiteRT GPU inference. Its native engines were already installed. NPU paths still need their own checks.',
    link: 'Build the Android SDK ↗',
  },
  linux: {
    status: 'PYTHON API · ARM64 CPU CHECKED', name: 'From development box to edge device.',
    description: 'Use the runtime inside your application, or start a host when another device needs access. Package model variants for the board’s actual SoC and installed engines.',
    boundary: 'Linux ARM64 CPU execution was exercised in a VM. VENTUNO Q’s QCS8275 NPU and camera path still require board validation. Native engine updates can use the device’s distribution system.',
    link: 'Read the IoT path ↗',
  },
  macos: {
    status: 'PYTHON API · APPLE SILICON CHECKED', name: 'Start on the Mac you already use.',
    description: 'Develop and run locally with the SDK. Export a supported PyTorch model to portable runtime packages, then add the target artifacts your users’ devices need.',
    boundary: 'ExecuTorch and LiteRT CPU inference and chat have been exercised. Those checks do not establish Metal or Core ML acceleration. A dedicated DTR desktop app is not required.',
    link: 'Try the native example ↗',
  },
  windows: {
    status: 'TARGET · NATIVE EXECUTION PENDING', name: 'Windows belongs in the same workflow.',
    description: 'Develop against the same application contract on a Windows laptop. Keep x64 CPU and Snapdragon ARM64/NPU artifacts as separate compatible targets.',
    boundary: 'Windows execution is pending. The next integration can use Windows ML’s managed execution providers where available. A Windows x64 run would not establish Snapdragon NPU support.',
    link: 'Read Windows preparation ↗',
  },
  web: {
    status: 'TYPESCRIPT SDK · WASM / WEBGPU', name: 'Deliver an engine with your web assets.',
    description: 'Use local LiteRT tensor inference or a compatible LiteRT-LM web model. Version the JS, WASM, and model together, then load a qualified release in a new session.',
    boundary: 'The Pixel’s embedded WebView executed a small model on WebGPU with LiteRT.js 2.5.2 and 2.5.3 while the APK stayed unchanged. This does not establish a browser-native vendor NPU plugin loader.',
    link: 'Read the web SDK path ↗',
  },
};
const tabs = [...document.querySelectorAll('[data-platform]')];
function select(tab) {
  const data = platforms[tab.dataset.platform];
  for (const item of tabs) {
    item.setAttribute('aria-selected', String(item === tab));
    item.tabIndex = item === tab ? 0 : -1;
  }
  document.getElementById('platform-panel').setAttribute('aria-labelledby', tab.id);
  for (const key of ['status', 'name', 'description', 'boundary', 'link']) {
    document.getElementById('platform-' + key).textContent = data[key];
  }
}
for (const tab of tabs) {
  tab.addEventListener('click', () => select(tab));
  tab.addEventListener('keydown', event => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const index = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1
      : (tabs.indexOf(tab) + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
    select(tabs[index]); tabs[index].focus();
  });
}
select(tabs[0]);
