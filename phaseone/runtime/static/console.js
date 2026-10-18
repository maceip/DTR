import {NativeHost, BrowserRuntime, base64FromBytes} from './sdk/index.js';

const $ = id => document.getElementById(id);
let host, browser, application, packages = [], info, hostDevice, session, running, imageUrl;
const labels = {executorch: 'ExecuTorch', litert: 'LiteRT'};
$('host-url').value = location.origin;

function clearError() { $('error').hidden = true; }
function fail(error) {
  $('error-title').textContent = (error.code || 'request_failed').replaceAll('_', ' ');
  $('error-message').textContent = error.message || String(error);
  $('error-extra').hidden = !error.details;
  $('error-details').textContent = error.details ? JSON.stringify(error.details, null, 2) : '';
  $('error').hidden = false;
  $('run-label').textContent = 'Request could not complete.';
}
function setBusy(value, message = 'Running…') {
  $('run-label').textContent = value ? message : 'Ready when you are.';
  for (const id of ['package', 'model', 'location', 'backend', 'new-session', 'send', 'classify', 'infer', 'disconnect']) $(id).disabled = value;
  $('cancel').hidden = !value;
}
function currentModel() { return info?.models[$('model').value]; }
function available(backend) {
  const model = currentModel();
  return model?.variants.some(variant => variant.backend === backend && !(info.compatibility?.[$('model').value]?.find(item => (item.variant || item.id) === variant.id)?.reasons?.length));
}
function option(value, text) { const element = document.createElement('option'); element.value = value; element.textContent = text; return element; }
function message(role, text) {
  $('conversation').querySelector('.empty')?.remove();
  const entry = document.createElement('div'); entry.className = 'message ' + role;
  const name = document.createElement('span'); name.className = 'role'; name.textContent = role === 'user' ? 'You' : role === 'assistant' ? 'Assistant' : 'Tool';
  const content = document.createElement('span'); content.className = 'content'; content.textContent = text;
  entry.append(name, content); $('conversation').append(entry); $('conversation').scrollTop = $('conversation').scrollHeight;
  return content;
}
function resetConversation() {
  session = undefined;
  $('conversation').replaceChildren();
  const empty = document.createElement('div'); empty.className = 'empty';
  const title = document.createElement('h3'); title.textContent = 'Same conversation. Different engine.';
  const help = document.createElement('p'); help.textContent = 'Send a message, switch engines, and keep the context.';
  empty.append(title, help); $('conversation').append(empty); $('session-label').textContent = 'New conversation';
}
function selection() { return {backend: $('backend').value, ...(running ? {signal: running.signal} : {})}; }
async function refreshDetails() {
  info = await application.describe();
  $('package-name').textContent = info.name + ' · ' + info.version;
  $('package-digest').textContent = info.digest;
  $('fact-location').textContent = $('location').value === 'browser' ? 'This browser' : `${hostDevice.system} · ${hostDevice.architecture}`;
  $('fact-engine').textContent = labels[$('backend').value];
  $('variants').replaceChildren();
  for (const item of info.compatibility?.[$('model').value] || []) {
    const element = document.createElement('div'); element.className = 'variant';
    const title = document.createElement('strong'); title.textContent = item.variant || item.id;
    const status = document.createElement('span'); status.className = item.reasons?.length ? 'unsupported' : 'supported';
    status.textContent = item.reasons?.length ? item.reasons.join(' · ') : item.execution_verified ? 'Executed here' : 'Compatible target · not run';
    element.append(title, status); $('variants').append(element);
  }
  for (const element of $('backend').options) element.disabled = !available(element.value);
  if (!available($('backend').value)) $('backend').value = available('executorch') ? 'executorch' : 'litert';
  $('fact-engine').textContent = labels[$('backend').value];
  const compatible = available($('backend').value);
  for (const id of ['send', 'classify', 'infer']) $(id).disabled = !compatible;
  if (!compatible) $('run-label').textContent = 'This model has no compatible variant here. Choose another location or model.';
}
async function chooseApplication(preserve = false) {
  clearError(); setBusy(true, 'Checking compatibility…');
  try {
    const history = preserve && session ? (await application.session(session.id)).messages : undefined;
    application = $('location').value === 'browser' ? await browser.application($('package').value) : await host.application($('package').value);
    session = history ? await application.createSession($('model').value, history) : undefined;
    if (!preserve) resetConversation();
    info = await application.describe();
    const model = currentModel(), chat = model.task === 'chat', vision = model.task === 'image.classification';
    $('chat-pane').hidden = !chat; $('image-pane').hidden = !vision; $('tensor-pane').hidden = chat || vision;
    $('task-title').textContent = chat ? 'Conversation' : vision ? 'Image classification' : 'Tensor inference';
    $('new-session').hidden = !chat;
    $('agent-control').hidden = !info.agent || info.agent.model !== $('model').value;
    $('agent-mode').checked = false;
    if (!chat && !vision) {
      const signatures = model.variants.find(variant => variant.signatures && Object.keys(variant.signatures).length)?.signatures || {};
      $('signature').replaceChildren(...Object.keys(signatures).map(name => option(name, name)));
      const signature = Object.values(signatures)[0];
      $('tensor-inputs').value = JSON.stringify(Object.fromEntries((signature?.inputs || []).map(input => {
        const shape = input.shape.map(value => value === -1 ? 1 : value), size = shape.reduce((a, b) => a * b, 1);
        return [input.name, {dtype: input.dtype, shape, data: Array(Math.min(size, 1024)).fill(1)}];
      })), null, 2);
    }
    $('execution-proof').className = 'proof'; $('execution-proof').textContent = 'Awaiting a successful run';
    $('fact-time').textContent = 'No run yet';
    $('execution-details').textContent = 'Run a model to see its actual execution metadata.';
    if (session) $('session-label').textContent = 'Context transferred to ' + ($('location').value === 'browser' ? 'browser' : 'native host');
  } finally { setBusy(false); if (application) await refreshDetails(); }
}
async function showExecution(result) {
  const execution = result.execution;
  $('execution-details').textContent = JSON.stringify({execution, ...(result.receipt ? {receipt: result.receipt} : {})}, null, 2);
  $('fact-engine').textContent = `${labels[execution.backend]} ${execution.runtime_version || ''}`.trim();
  $('fact-location').textContent = execution.location === 'browser' ? 'This browser' : 'Paired native host';
  const duration = execution.inference_ms ?? execution.elapsed_ms;
  $('fact-time').textContent = Number.isFinite(duration) ? `${duration.toFixed(1)} ms` : 'Complete';
  $('execution-proof').className = 'proof verified';
  $('execution-proof').textContent = `Executed · ${execution.variant} · ${execution.requested_accelerator || 'cpu'}`;
  await refreshDetails();
}
async function run(callback) {
  clearError(); running = new AbortController(); setBusy(true, 'Loading model and running…');
  try { await callback(); $('run-label').textContent = 'Complete.'; }
  catch (error) { fail(error); }
  finally { const status = $('run-label').textContent; running = undefined; setBusy(false); await refreshDetails().catch(fail); $('run-label').textContent = status; }
}
$('pair-form').addEventListener('submit', async event => {
  event.preventDefault(); clearError(); const button = event.submitter; button.disabled = true;
  try {
    const url = $('host-url').value.replace(/\/$/, '');
    const health = await fetch(url + '/health', {signal: AbortSignal.timeout(5000)}).then(response => response.json());
    if (health.product !== 'PHASEONE Runtime') throw new Error('This address is not a PHASEONE runtime host');
    host = new NativeHost(url, $('host-token').value.trim());
    hostDevice = await host.device(); packages = await host.packages();
    if (!packages.length) throw new Error('The host has no application packages. Start it with --package PATH.');
    browser = new BrowserRuntime(host);
    $('host-token').value = '';
    $('host-details').textContent = JSON.stringify(hostDevice, null, 2);
    $('package').replaceChildren(...packages.map(pkg => option(pkg.digest, pkg.name)));
    const assistant = packages.find(pkg => Object.values(pkg.models).some(model => model.task === 'chat'));
    if (assistant) $('package').value = assistant.digest;
    fillModels(); await chooseApplication();
    $('pairing').hidden = true; $('workspace').hidden = false; $('disconnect').hidden = false;
    $('connection-status').classList.add('connected');
    $('connection-status').replaceChildren(Object.assign(document.createElement('span'), {className: 'dot'}), document.createTextNode(`${hostDevice.system} · ${hostDevice.architecture} paired`));
  } catch (error) { fail(error); }
  finally { button.disabled = false; }
});
function fillModels() { const pkg = packages.find(pkg => pkg.digest === $('package').value); $('model').replaceChildren(...Object.keys(pkg.models).map(name => option(name, name))); }
$('package').addEventListener('change', () => { fillModels(); chooseApplication().catch(fail); });
$('model').addEventListener('change', () => chooseApplication().catch(fail));
$('location').addEventListener('change', () => chooseApplication(true).catch(fail));
$('backend').addEventListener('change', () => { clearError(); refreshDetails().catch(fail); });
$('new-session').addEventListener('click', resetConversation);
$('disconnect').addEventListener('click', async () => { await browser?.close(); location.reload(); });
$('cancel').addEventListener('click', () => running?.abort());
$('chat-form').addEventListener('submit', event => {
  event.preventDefault(); const prompt = $('prompt').value.trim(); if (!prompt) return;
  run(async () => {
    if (!session) session = await application.createSession($('model').value, [{role: 'system', content: 'You are a concise assistant. Answer in one short sentence. Keep facts about the user separate from facts about yourself.'}]);
    message('user', prompt); $('prompt').value = ''; const output = message('assistant', '');
    for await (const event of application.sendStream(session.id, prompt, {...selection(), agent: $('agent-mode').checked, generation: {max_output_tokens: 128, timeout_seconds: 120}})) {
      if (event.type === 'start') $('run-label').textContent = `Generating with ${labels[event.execution.backend]}…`;
      if (event.type === 'delta') { output.textContent += event.text; $('conversation').scrollTop = $('conversation').scrollHeight; }
      if (event.type === 'tool_result') message('tool', `${event.action.tool || 'Tool'} · ${event.action.status}`);
      if (event.type === 'done') { output.textContent = typeof event.message.content === 'string' ? event.message.content : JSON.stringify(event.message.content); await showExecution(event); }
    }
    $('session-label').textContent = 'Session ' + session.id.slice(0, 8);
  }).catch(fail);
});
$('image-file').addEventListener('change', () => {
  if (imageUrl) URL.revokeObjectURL(imageUrl);
  const file = $('image-file').files[0]; $('image-preview').hidden = !file;
  if (file) { imageUrl = URL.createObjectURL(file); $('image-preview').src = imageUrl; }
  $('predictions').replaceChildren();
});
$('classify').addEventListener('click', () => run(async () => {
  const file = $('image-file').files[0]; if (!file) throw new Error('Choose an image first.');
  const input = $('location').value === 'browser' ? file : base64FromBytes(new Uint8Array(await file.arrayBuffer()));
  const result = await application.classify($('model').value, input, {...selection(), top_k: 5});
  $('predictions').replaceChildren(...result.predictions.map(prediction => { const row = document.createElement('li'), label = document.createElement('span'), score = document.createElement('span'); label.textContent = prediction.label; score.textContent = (prediction.score * 100).toFixed(1) + '%'; row.append(label, score); return row; }));
  await showExecution(result);
}).catch(fail));
$('infer').addEventListener('click', () => run(async () => {
  const result = await application.infer($('model').value, JSON.parse($('tensor-inputs').value), {...selection(), signature: $('signature').value});
  $('tensor-result').hidden = false; $('tensor-result').textContent = JSON.stringify(result.outputs, null, 2); await showExecution(result);
}).catch(fail));
